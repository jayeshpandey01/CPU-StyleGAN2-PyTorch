#!/usr/bin/env python3
"""
StyleGAN2 Diagnostics Tool

This script provides diagnostic utilities for StyleGAN2 models to help identify
and troubleshoot issues with model loading, weight conversion, and face generation.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict

# Add current directory to path
sys.path.append('.')

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = 'cpu'

def check_model_files():
    """Check for the existence of required model files"""
    print("\n=== StyleGAN2 Model Files ===")
    
    model_files = [
        'model.py',
        'stylegan2-ffhq-config-f.pkl',
        'op/__init__.py',
        'op/fused_act.py',
        'op/upfirdn2d.py',
        'cpu_ops.py'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✓ Found {file_path}")
        else:
            print(f"✗ Missing {file_path}")
    
    # Check for converted weights
    if os.path.exists('stylegan2-ffhq-config-f_pt.pkl'):
        print("✓ Found converted PyTorch weights")
    else:
        print("✗ Missing converted PyTorch weights")


def analyze_weights_file(model_path):
    """Analyze the structure of a StyleGAN2 weights file"""
    print(f"\n=== Analyzing {model_path} ===")
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        if isinstance(data, dict):
            print("File contains a dictionary with keys:")
            for key in data.keys():
                print(f"  - {key}")
                
            # Check for TensorFlow vs PyTorch format
            if 'g' in data:
                print("\nThis appears to be a TensorFlow format weights file.")
                tf_keys = list(data['g'].keys())
                print(f"Found {len(tf_keys)} TensorFlow weight keys.")
                print("Sample keys:")
                for key in tf_keys[:10]:  # Show first 10 keys
                    val = data['g'][key]
                    if isinstance(val, np.ndarray):
                        print(f"  - {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"  - {key}: {type(val)}")
            
            elif 'g_ema' in data:
                print("\nThis appears to be a PyTorch format weights file.")
                pt_keys = list(data['g_ema'].keys())
                print(f"Found {len(pt_keys)} PyTorch weight keys.")
                print("Sample keys:")
                for key in pt_keys[:10]:  # Show first 10 keys
                    val = data['g_ema'][key]
                    if isinstance(val, torch.Tensor):
                        print(f"  - {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"  - {key}: {type(val)}")
            else:
                print("\nUnknown weights format, neither PyTorch nor TensorFlow keys found.")
        else:
            print(f"File contains a {type(data)} instead of a dictionary.")
    
    except Exception as e:
        print(f"Error analyzing weights file: {str(e)}")
        import traceback
        traceback.print_exc()


def patch_model_for_cpu():
    """Apply CPU-compatible patches to the model"""
    print("\n=== Patching Model for CPU ===")
    
    try:
        import model
        
        # Check if cpu_ops.py exists
        if os.path.exists('cpu_ops.py'):
            from cpu_ops import (
                cpu_conv2d, 
                cpu_conv_transpose2d,
                simplified_upfirdn2d,
                SafeNoiseInjection,
                SafeToRGB
            )
            
            # Apply patches
            model.conv2d_gradfix.conv2d = cpu_conv2d
            model.conv2d_gradfix.conv_transpose2d = cpu_conv_transpose2d
            model.upfirdn2d = simplified_upfirdn2d
            model.NoiseInjection = SafeNoiseInjection
            model.ToRGB = SafeToRGB
            
            print("✓ Successfully patched model with CPU operations")
        else:
            print("✗ cpu_ops.py not found, creating simplified patches...")
            
            # Create simplified patches
            def cpu_conv2d(x, weight, bias=None, stride=1, padding=0, groups=1):
                return torch.nn.functional.conv2d(x, weight, bias, stride, padding, groups=groups)
                
            def cpu_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
                return torch.nn.functional.conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, dilation)
            
            def simplified_upfirdn2d(input, kernel=None, up=1, down=1, pad=(0, 0)):
                if up > 1:
                    input = torch.nn.functional.interpolate(input, scale_factor=up, mode='nearest')
                return input
            
            # Apply simplified patches
            model.conv2d_gradfix.conv2d = cpu_conv2d
            model.conv2d_gradfix.conv_transpose2d = cpu_conv_transpose2d
            model.upfirdn2d = simplified_upfirdn2d
            
            print("✓ Applied simplified CPU patches")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to patch model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(model_path, size=128, channel_multiplier=1):
    """Test loading the StyleGAN2 model"""
    print(f"\n=== Testing Model Loading (size={size}) ===")
    
    try:
        from model import Generator
        
        # Initialize the model
        print("Initializing model...")
        g_ema = Generator(
            size, 512, 8, channel_multiplier=channel_multiplier, blur_kernel=[1, 3, 3, 1]
        ).to(device)
        
        # Count model parameters
        num_params = sum(p.numel() for p in g_ema.parameters())
        print(f"Model has {num_params:,} parameters")
        
        # Try to load weights
        print(f"Loading weights from {model_path}...")
        
        # First try converted weights
        converted_path = model_path.replace('.pkl', '_pt.pkl')
        if os.path.exists(converted_path):
            print(f"Found converted weights at {converted_path}")
            state_dict = torch.load(converted_path)
            g_ema.load_state_dict(state_dict['g_ema'], strict=False)
            print("✓ Loaded converted PyTorch weights")
        else:
            # Try direct loading
            with open(model_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            if isinstance(data, dict) and 'g_ema' in data:
                g_ema.load_state_dict(data['g_ema'], strict=False)
                print("✓ Loaded PyTorch weights directly")
            elif isinstance(data, dict) and 'g' in data:
                print("Found TensorFlow weights, conversion needed")
                
                # Try to use conversion utility
                if os.path.exists('convert_tf_to_pt.py'):
                    from convert_tf_to_pt import convert_tf_to_pt
                    
                    print("Converting TensorFlow weights...")
                    converted = convert_tf_to_pt(data['g'], g_ema)
                    g_ema.load_state_dict(converted['g_ema'], strict=False)
                    print("✓ Converted and loaded TensorFlow weights")
                else:
                    print("✗ convert_tf_to_pt.py not found, can't convert weights")
            else:
                print("✗ Unrecognized weights format")
        
        # Compute mean latent vector
        print("Computing mean latent vector...")
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(32)  # Use fewer samples for diagnostics
            
        print(f"Mean latent shape: {mean_latent.shape}")
        print(f"Mean latent stats: min={mean_latent.min().item():.4f}, max={mean_latent.max().item():.4f}")
        print(f"Mean latent mean={mean_latent.mean().item():.4f}, std={mean_latent.std().item():.4f}")
        
        # Check if mean latent looks reasonable (non-random)
        if mean_latent.std().item() < 0.05:
            print("✓ Mean latent looks properly initialized (low standard deviation)")
        else:
            print("✗ Mean latent has high standard deviation, may indicate random initialization")
        
        return g_ema, mean_latent
        
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_face_generation(g_ema, mean_latent, seed=42):
    """Test face generation with the loaded model"""
    print("\n=== Testing Face Generation ===")
    
    if g_ema is None or mean_latent is None:
        print("✗ Cannot test face generation, model not loaded")
        return None
    
    try:
        # Fix random seed for reproducibility
        torch.manual_seed(seed)
        
        # Generate latent vector
        latent_z = torch.randn(1, 512, device=device)
        
        print("Processing latent through style network...")
        latent_w = g_ema.get_latent(latent_z)
        
        # Apply truncation trick
        truncation = 0.7
        latent_w = mean_latent + truncation * (latent_w - mean_latent)
        
        # Prepare for synthesis
        latent_w = latent_w.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        
        # Create noise
        print("Generating noise...")
        noise = []
        layer_count = int(np.log2(g_ema.size)) * 2 - 2
        
        current_size = 4
        for i in range(layer_count):
            if i % 2 == 0 and i > 0:
                current_size *= 2
                
            noise.append(torch.randn(1, 1, current_size, current_size, device=device) * 0.1)
            
        print(f"Created {len(noise)} noise tensors")
        
        # Generate image
        print("Generating face...")
        with torch.no_grad():
            g_ema.eval()
            img, _ = g_ema(
                [latent_w],
                noise=noise,
                input_is_latent=True,
                truncation=truncation,
                truncation_latent=mean_latent,
                randomize_noise=False
            )
        
        # Check image
        print(f"Generated image shape: {img.shape}")
        print(f"Image range: [{img.min().item():.4f}, {img.max().item():.4f}]")
        print(f"Image mean: {img.mean().item():.4f}, std: {img.std().item():.4f}")
        
        # Normalize and convert to numpy for display
        img_np = (img[0].cpu().permute(1, 2, 0).numpy() + 1) * 0.5
        img_np = np.clip(img_np, 0, 1)
        
        # Check if image looks reasonable (non-random)
        if img.std().item() > 0.2 and abs(img.mean().item()) < 0.5:
            print("✓ Image statistics look reasonable")
        else:
            print("✗ Image statistics suggest possible random output")
        
        # Save the image
        os.makedirs('diagnostics', exist_ok=True)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_path = f'diagnostics/test_face_seed{seed}.png'
        img_pil.save(img_path)
        print(f"✓ Test face saved to {img_path}")
        
        # Plot image with histogram for diagnostics
        plt.figure(figsize=(12, 6))
        
        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title('Generated Face')
        plt.axis('off')
        
        # Show histogram
        plt.subplot(1, 2, 2)
        plt.hist(img_np.reshape(-1), bins=50, alpha=0.7, color='blue')
        plt.title('Pixel Value Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        # Save the diagnostic plot
        plt.tight_layout()
        plt.savefig('diagnostics/test_face_diagnostics.png')
        print("✓ Diagnostic plot saved to diagnostics/test_face_diagnostics.png")
        
        return img_np
        
    except Exception as e:
        print(f"✗ Failed to generate face: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def convert_weights_if_needed(model_path):
    """Convert TensorFlow weights to PyTorch format if needed"""
    print("\n=== Weight Conversion Check ===")
    
    converted_path = model_path.replace('.pkl', '_pt.pkl')
    
    if os.path.exists(converted_path):
        print(f"✓ Converted weights already exist at {converted_path}")
        return converted_path
    
    # Check if this is a TensorFlow model that needs conversion
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        if isinstance(data, dict) and 'g' in data and 'g_ema' not in data:
            print("Found TensorFlow weights, conversion needed")
            
            # Try to use conversion utilities
            if os.path.exists('convert_tf_to_pt.py'):
                from convert_tf_to_pt import convert_tf_to_pt
                
                print("Converting TensorFlow weights to PyTorch format...")
                from model import Generator
                dummy_model = Generator(1024, 512, 8)  # Use full size for conversion
                converted = convert_tf_to_pt(data['g'], dummy_model)
                
                # Save converted weights
                torch.save(converted, converted_path)
                print(f"✓ Converted weights saved to {converted_path}")
                return converted_path
            else:
                print("✗ convert_tf_to_pt.py not found, can't convert weights")
                return None
        else:
            print("✓ Weights already in PyTorch format, no conversion needed")
            return model_path
    
    except Exception as e:
        print(f"✗ Error checking/converting weights: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_full_diagnostics(model_path, size=128, channel_multiplier=1):
    """Run all diagnostic tests"""
    print("=" * 50)
    print("StyleGAN2 Diagnostics Tool")
    print("=" * 50)
    
    # Check system info
    print("\n=== System Info ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    
    # Check files
    check_model_files()
    
    # Analyze weights file
    analyze_weights_file(model_path)
    
    # Convert weights if needed
    convert_weights_if_needed(model_path)
    
    # Patch model for CPU
    patch_model_for_cpu()
    
    # Test model loading
    g_ema, mean_latent = test_model_loading(model_path, size, channel_multiplier)
    
    # Test face generation
    if g_ema is not None and mean_latent is not None:
        test_face_generation(g_ema, mean_latent)
    
    print("\n" + "=" * 50)
    print("Diagnostics Complete")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StyleGAN2 Diagnostics Tool')
    parser.add_argument('--model', type=str, default='stylegan2-ffhq-config-f.pkl',
                        help='Path to the StyleGAN2 model pickle')
    parser.add_argument('--size', type=int, default=128,
                        help='Output image size (reduced for CPU efficiency)')
    parser.add_argument('--channel-multiplier', type=int, default=1,
                        help='Channel multiplier (use 1 for CPU efficiency)')
    args = parser.parse_args()
    
    run_full_diagnostics(args.model, args.size, args.channel_multiplier)