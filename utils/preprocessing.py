
import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def preprocess_for_model(self, image):
        """
        Final preprocessing for model input
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif hasattr(image, 'cpu'):  # Handle torch tensors
            image = image.cpu().numpy()
        
        # Ensure correct dtype
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Resize
        image = cv2.resize(image, self.target_size)
        
        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def combine_with_features(self, image, fertilizer_amount, days):
        """
        Combine image with numerical features
        """
        features = np.array([fertilizer_amount, days])
        return image, features
