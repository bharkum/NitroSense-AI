"""
Setup script for U2Net
Run this once before deploying to download U2Net
"""
import os
import urllib.request
import zipfile
import shutil

def setup_u2net():
    """Download and setup U2Net"""
    
    # Create directories
    os.makedirs('u2net/model', exist_ok=True)
    os.makedirs('u2net/weights', exist_ok=True)
    
    # Download U2Net source
    print("📥 Downloading U2Net source...")
    u2net_url = "https://github.com/xuebinqin/U-2-Net/archive/master.zip"
    zip_path = "u2net/u2net_source.zip"
    
    try:
        urllib.request.urlretrieve(u2net_url, zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("u2net/")
        
        # Copy model files
        source_model = os.path.join("u2net", "U-2-Net-master", "model")
        if os.path.exists(source_model):
            for file in os.listdir(source_model):
                if file.endswith('.py'):
                    shutil.copy(
                        os.path.join(source_model, file),
                        os.path.join("u2net", "model", file)
                    )
        
        # Clean up
        os.remove(zip_path)
        shutil.rmtree(os.path.join("u2net", "U-2-Net-master"))
        
        print("
✅ U2Net source setup complete!")
        print("
📌 Next step:")
        print("1. Download u2net.pth weights from:")
        print("   https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ")
        print("2. Place it in: u2net/weights/u2net.pth")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    setup_u2net()
