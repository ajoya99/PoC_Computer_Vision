"""Download RT-DETR-Medium model weights."""

import urllib.request
from pathlib import Path
import subprocess

if __name__ == "__main__":
    print("Attempting to download RT-DETR-Medium model using Ultralytics...")
    
    try:
        # Try using yolo CLI to download model
        result = subprocess.run(
            ["yolo", "detect", "predict", "model=rtdetr-m.pt", "source=https://ultralytics.com/images/bus.jpg"],
            capture_output=True,
            text=True,
            timeout=60
        )
        print("Model download initiated via YOLO CLI")
        if result.returncode == 0:
            print("✓ Model downloaded successfully")
        else:
            print(f"Warning: {result.stderr}")
    except Exception as e:
        print(f"CLI approach failed: {e}")
        
    # Alternative: Try direct download from Ultralytics models
    print("\nTrying direct URL approach...")
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/rtdetr-m.pt",
        "https://ultralytics.com/assets/rtdetr-m.pt",
        "https://api.roboflow.com/models/rtdetr-m/1/yolov8/rtdetr-m.pt",
    ]
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, "rtdetr-m.pt", reporthook=lambda blocknum, blocksize, totalsize: print(f"Downloaded: {blocknum * blocksize / (1024*1024):.1f} MB"))
            size_mb = Path("rtdetr-m.pt").stat().st_size / (1024 * 1024)
            print(f"✓ Successfully downloaded rtdetr-m.pt ({size_mb:.1f} MB)")
            break
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\n✗ All download methods failed")
        print("Please manually download from: https://github.com/ultralytics/assets/releases")
