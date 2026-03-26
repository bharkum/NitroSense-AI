"""
Portable U2Net Background Remover
Works on any system - downloads model if not present
Memory-optimized for web applications
"""
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import sys
import os
import gc
import urllib.request
import zipfile
import shutil
import warnings
warnings.filterwarnings('ignore')

# Memory settings
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class BackgroundRemover:
    """
    Portable U2Net Background Remover
    Automatically downloads U2Net if not found
    Memory-optimized for web applications
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device('cpu')
        
        # Get paths
        self.current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
        self.project_root = os.path.dirname(self.current_dir) if '__file__' in dir() else self.current_dir
        
        # Set up paths
        self.u2net_dir = os.path.join(self.project_root, 'u2net')
        self.model_dir = os.path.join(self.u2net_dir, 'model')
        self.weights_dir = os.path.join(self.u2net_dir, 'weights')
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Download/verify U2Net source
        self.u2net_available = self._ensure_u2net_source()
        
        # Set model path
        if model_path is None:
            model_path = os.path.join(self.weights_dir, 'u2net.pth')
        self.model_path = model_path
        
        # Import and load model
        self._import_u2net()
        self._load_model()
        
        # Training transform
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        gc.collect()
        if self.model is not None:
            print("✅ BackgroundRemover initialized with U2Net model")
        else:
            print("✅ BackgroundRemover initialized in fallback mode")
    
    def _ensure_u2net_source(self):
        """Download U2Net source if not present"""
        u2net_py = os.path.join(self.model_dir, 'u2net.py')
        
        if not os.path.exists(u2net_py):
            print("📥 Downloading U2Net source...")
            u2net_url = "https://github.com/xuebinqin/U-2-Net/archive/master.zip"
            zip_path = os.path.join(self.u2net_dir, 'u2net.zip')
            
            try:
                # Download
                urllib.request.urlretrieve(u2net_url, zip_path)
                
                # Extract
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.u2net_dir)
                
                # Move model files
                extracted = os.path.join(self.u2net_dir, 'U-2-Net-master')
                if os.path.exists(extracted):
                    src_model = os.path.join(extracted, 'model')
                    if os.path.exists(src_model):
                        for file in os.listdir(src_model):
                            if file.endswith('.py'):
                                shutil.copy(
                                    os.path.join(src_model, file),
                                    os.path.join(self.model_dir, file)
                                )
                
                # Clean up
                os.remove(zip_path)
                if os.path.exists(extracted):
                    shutil.rmtree(extracted)
                
                print("✅ U2Net source downloaded successfully")
                return True
                
            except Exception as e:
                print(f"⚠️ Could not download U2Net: {e}")
                return False
        else:
            print("✅ U2Net source already present")
            return True
    
    def _import_u2net(self):
        """Import U2Net module"""
        if self.model_dir not in sys.path:
            sys.path.insert(0, self.model_dir)
        
        try:
            from u2net import U2NET
            self.U2NET = U2NET
            self.u2net_available = True
        except ImportError:
            self.U2NET = None
            self.u2net_available = False
    
    def _load_model(self):
        """Load the U2Net model"""
        if self.U2NET is None:
            self.model = None
            return
            
        if not os.path.exists(self.model_path):
            self.model = None
            return
        
        try:
            self.model = self.U2NET(in_ch=3, out_ch=1)
            state_dict = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.model = None
    
    def remove_background(self, image, target_size=(224, 224), max_size=800):
        """
        Memory-optimized background removal - preserves original leaf colors
        
        Args:
            image: numpy array or PIL Image
            target_size: final size for model input
            max_size: maximum dimension for processing
            
        Returns:
            result: numpy array with background removed (original colors preserved)
            mask: binary mask from U2Net
        """
        # Fallback if model not loaded
        if self.model is None:
            if isinstance(image, np.ndarray):
                return cv2.resize(image, target_size), None
            elif isinstance(image, Image.Image):
                return np.array(image.resize(target_size)), None
            return image, None
        
        try:
            # Store original image for color preservation
            original_image = image.copy() if isinstance(image, np.ndarray) else image
            
            # Resize large images for processing
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, Image.Image):
                if max(image.size) > max_size:
                    scale = max_size / max(image.size)
                    new_size = tuple(int(dim * scale) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    image = image
            
            # Get mask
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                d1, d2, d3, d4, d5, d6, d7 = self.model(input_tensor)
            
            mask = d1[:, 0, :, :].cpu().numpy()
            
            # Clean up
            del d1, d2, d3, d4, d5, d6, d7, input_tensor
            gc.collect()
            
            # Process mask
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            mask = mask.transpose(1, 2, 0)
            mask_target = cv2.resize(mask, target_size)
            mask_target = (mask_target * 255).astype(np.uint8)
            
            # === FIX: Preserve original colors ===
            # Get the original image resized to target size
            if isinstance(original_image, np.ndarray):
                image_resized = cv2.resize(original_image, target_size)
            else:
                image_resized = np.array(original_image.resize(target_size, Image.Resampling.LANCZOS))
            
            # Create result by copying the resized original
            result = image_resized.copy()
            
            # Handle mask dimensions
            if len(mask_target.shape) == 3 and mask_target.shape[2] == 1:
                mask_target = mask_target.squeeze()
            
            # Set background to white
            result[mask_target < 128] = [255, 255, 255]
            
            return result, mask_target
            
        except Exception as e:
            print(f"Warning: Background removal failed - {e}")
            if isinstance(image, Image.Image):
                return np.array(image.resize(target_size)), None
            return cv2.resize(image, target_size), None
    
    def __del__(self):
        try:
            gc.collect()
        except:
            pass
