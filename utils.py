"""
Utility module containing directories initialization, download utilities, and evaluation metrics.
"""
import os
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import config

def initialize_directories():
    """Initializes required directory structures for caching assets."""
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    os.makedirs(config.EMB_DIR, exist_ok=True)
    
    print("[INFO] CONFIG SETUP DONE")
    print(f"TRAIN: {config.TRAIN_CSV}")
    print(f"TEST: {config.TEST_CSV}")
    print(f"IMAGE_DIR: {config.IMAGE_DIR}")
    print(f"EMB_DIR: {config.EMB_DIR}")



def download_image(url):
    """
    Downloads an image from a remote URL and converts it to standard RGB color space.
    """
    try:
        response = requests.get(url, timeout=10)
        image_asset = Image.open(BytesIO(response.content)).convert('RGB')
        return image_asset
    except Exception:
        # Gracefully handle download issues (network timeout, broken links)
        return None
