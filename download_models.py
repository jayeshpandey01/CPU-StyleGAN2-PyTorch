# Utility to download StyleGAN2 pretrained models directly
import os
import sys
import requests
import gdown
import zipfile
import hashlib

# Available models
AVAILABLE_MODELS = {
    "ffhq": "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl",
    "ffhq-config-e": "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-e.pkl",
    "church": "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-f.pkl",
    "cat": "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl",
    "horse": "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-f.pkl",
    "car": "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl",
    "ada-ffhq": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
    "ada-metfaces": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
    "stylegan3-t": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl",
    "stylegan3-r": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl",
}

def download_model(model_name, output_dir="models"):
    """Download a pretrained StyleGAN2/3 model

    Args:
        model_name: Name of the model to download (see AVAILABLE_MODELS)
        output_dir: Directory to save the model
    
    Returns:
        str: Path to the downloaded model
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model URL
    url = AVAILABLE_MODELS[model_name]
    
    # Output file path
    output_path = os.path.join(output_dir, f"{model_name}.pkl")
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"Model {model_name} already downloaded at {output_path}")
        return output_path
    
    print(f"Downloading {model_name} model from {url}...")
    
    # Download the file
    try:
        # Download with requests
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write file to disk
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            
            print(f"Successfully downloaded {model_name} to {output_path}")
            return output_path
        else:
            print(f"Failed to download {model_name}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        
        # Try using gdown as a fallback (for Google Drive links)
        try:
            print("Trying alternative download method...")
            gdown.download(url, output_path, quiet=False)
            if os.path.exists(output_path):
                print(f"Successfully downloaded {model_name} to {output_path}")
                return output_path
        except Exception as e2:
            print(f"Alternative download failed: {e2}")
        
        return None

def download_all_models(output_dir="models"):
    """Download all available pretrained models"""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in AVAILABLE_MODELS:
        print(f"Downloading {model_name}...")
        download_model(model_name, output_dir)
        print("-" * 40)

def verify_checksum(file_path, expected_md5):
    """Verify the MD5 checksum of a downloaded file"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read file in chunks
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    computed_md5 = md5_hash.hexdigest()
    if computed_md5 != expected_md5:
        print(f"Checksum mismatch for {file_path}")
        print(f"Expected: {expected_md5}")
        print(f"Got: {computed_md5}")
        return False
        
    print(f"Checksum verified for {file_path}")
    return True

# Run the script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pretrained StyleGAN2/3 models")
    parser.add_argument("--model", "-m", type=str, choices=list(AVAILABLE_MODELS.keys()), 
                        default="ffhq", help="Model to download")
    parser.add_argument("--output", "-o", type=str, default="models",
                        help="Output directory")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Download all available models")
    
    args = parser.parse_args()
    
    if args.all:
        download_all_models(args.output)
    else:
        download_model(args.model, args.output)
