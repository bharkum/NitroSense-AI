import os
import requests
import sys

def download_file(url, filename, description):
    print(f"📥 Downloading {description}...")
    try:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb = downloaded / (1024*1024)
                        mb_total = total_size / (1024*1024)
                        sys.stdout.write(f"\r   Progress: {percent:.1f}% ({mb:.1f}/{mb_total:.1f} MB)")
                        sys.stdout.flush()
        print(f"\n✅ Downloaded to {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def download_model():
    """Download model from Hugging Face"""
    # You'll need to upload your model to Hugging Face or another URL
    # For now, create a placeholder or skip
    print("⚠️ Model file needs to be uploaded separately")
    print("   Please upload Pyramid_fusion_densenet121_model.h5 to Hugging Face or Google Drive")
    return True

def download_u2net():
    """Download U2Net weights from official source"""
    url = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"
    filename = "u2net/weights/u2net.pth"
    return download_file(url, filename, "U2Net weights (168 MB)")

if __name__ == "__main__":
    print("="*50)
    print("Downloading large files for NitroSense AI")
    print("="*50)
    download_u2net()
    download_model()