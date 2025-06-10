import os
import sys
import subprocess
import torch
import shutil

def setup_stylegan2_ada():
    """
    Set up StyleGAN2-ADA repository for better weight loading
    """
    print("Setting up StyleGAN2-ADA dependencies...")
    
    # Check if we already have stylegan2-ada-pytorch folder
    if os.path.exists("stylegan2-ada-pytorch"):
        print("StyleGAN2-ADA folder already exists")
    else:
        print("Cloning StyleGAN2-ADA repository...")
        subprocess.run([
            "git", "clone", "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
        ], check=True)
    
    # Add to Python path
    stylegan2_path = os.path.abspath("stylegan2-ada-pytorch")
    if stylegan2_path not in sys.path:
        sys.path.append(stylegan2_path)
        print(f"Added {stylegan2_path} to Python path")
    
    # Check if we can import the modules
    try:
        sys.path.insert(0, stylegan2_path)
        import dnnlib
        import legacy
        print("Successfully imported StyleGAN2-ADA modules")
    except ImportError as e:
        print(f"Error importing StyleGAN2-ADA modules: {e}")
        print("You may need to install additional dependencies")
        
    return stylegan2_path

def download_ffhq_model():
    """
    Download the FFHQ model if not already present
    """
    os.makedirs("pretrained", exist_ok=True)
    
    if os.path.exists("pretrained/ffhq.pkl"):
        print("FFHQ model already downloaded")
        return "pretrained/ffhq.pkl"
        
    print("Downloading FFHQ model (this might take a while)...")
    import requests
    
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open("pretrained/ffhq.pkl", 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print("Downloaded FFHQ model successfully")
        return "pretrained/ffhq.pkl"
    else:
        print(f"Failed to download: {response.status_code}")
        return None

def convert_tf_to_local_format(model_path):
    """
    Convert TensorFlow .pkl to local compatible format
    """
    import pickle
    import numpy as np
    
    print(f"Converting {model_path} to local format...")
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create compatible state dictionary
    if 'g_ema' not in data:
        # Handle TensorFlow format
        tf_state_dict = data['g']
        state_dict = {}
        
        # Mapping network
        for i in range(8):
            key_prefix = f'g_mapping/Dense{i}'
            if f'{key_prefix}/weight' in tf_state_dict:
                state_dict[f'style.{i+1}.weight'] = torch.from_numpy(
                    tf_state_dict[f'{key_prefix}/weight'].transpose()
                )
                state_dict[f'style.{i+1}.bias'] = torch.from_numpy(
                    tf_state_dict[f'{key_prefix}/bias']
                )
        
        # Save converted weights
        torch.save({'g_ema': state_dict}, 'pretrained/converted_model.pt')
        print("Saved converted model to pretrained/converted_model.pt")
        return 'pretrained/converted_model.pt'
    
    return model_path

if __name__ == "__main__":
    # Setup the environment
    stylegan2_path = setup_stylegan2_ada()
    print(f"StyleGAN2-ADA path: {stylegan2_path}")
    
    # Try to download the model
    try:
        model_path = download_ffhq_model()
        if model_path:
            print(f"Model path: {model_path}")
            
            # Convert if needed
            converted_path = convert_tf_to_local_format(model_path)
            print(f"Ready to use model: {converted_path}")
    except Exception as e:
        print(f"Error setting up model: {e}")
