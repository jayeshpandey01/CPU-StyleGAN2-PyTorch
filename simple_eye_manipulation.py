"""
Face Image Manipulation using Simple Processing
(Fallback if StyleGAN2 doesn't work)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Check if we need to download sample faces
SAMPLE_FACES_URL = "https://github.com/NVlabs/ffhq-dataset/raw/master/thumbnails128x128/"

def download_sample_face(index=0, output_dir="sample_faces"):
    """Download a sample face from FFHQ dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Format index to 5 digits
    idx_str = f"{index:05d}"
    output_path = os.path.join(output_dir, f"{idx_str}.png")
    
    # Check if file exists
    if os.path.exists(output_path):
        print(f"Sample face already exists at {output_path}")
        return output_path
        
    # Download the file
    url = f"{SAMPLE_FACES_URL}{idx_str}.png"
    try:
        print(f"Downloading sample face from {url}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded sample face to {output_path}")
            return output_path
        else:
            print(f"Failed to download face. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading face: {e}")
        return None

def download_multiple_faces(start_idx=0, count=10, output_dir="sample_faces"):
    """Download multiple sample faces"""
    paths = []
    for i in range(start_idx, start_idx + count):
        path = download_sample_face(i, output_dir)
        if path:
            paths.append(path)
    return paths

def load_face_image(path):
    """Load a face image from path"""
    img = Image.open(path)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

def manipulate_eyes_simple(img_array, operation="enlarge", strength=0.5):
    """
    Simple eye manipulation using image processing techniques
    
    Args:
        img_array: Input image as numpy array (H,W,C), normalized to [0,1]
        operation: Type of manipulation ("enlarge", "close", "color", "move")
        strength: Strength of effect (0.0 to 1.0)
        
    Returns:
        np.ndarray: Modified image
    """
    # For simplicity, we'll focus on the eye region using approximate coordinates
    h, w = img_array.shape[:2]
    
    # Assuming face is centered, eyes are approximately at these coordinates
    # These are approximate values for 128x128 face images
    left_eye_y = int(h * 0.4)
    right_eye_y = int(h * 0.4)
    left_eye_x = int(w * 0.35)
    right_eye_x = int(w * 0.65)
    eye_h = int(h * 0.1)
    eye_w = int(w * 0.1)
    
    # Create a copy of the image to modify
    modified = img_array.copy()
    
    # Define eye regions
    left_eye_region = (
        max(0, left_eye_y - eye_h), 
        min(h, left_eye_y + eye_h), 
        max(0, left_eye_x - eye_w), 
        min(w, left_eye_x + eye_w)
    )
    
    right_eye_region = (
        max(0, right_eye_y - eye_h), 
        min(h, right_eye_y + eye_h), 
        max(0, right_eye_x - eye_w), 
        min(w, right_eye_x + eye_w)
    )
    
    # Apply different operations
    if operation == "enlarge":
        # Simple enlargement by taking a smaller region and stretching it
        scale = 1.0 - strength * 0.3  # Scale factor for cropping (0.7 to 1.0)
        
        # Left eye
        ly1, ly2, lx1, lx2 = left_eye_region
        lh, lw = ly2 - ly1, lx2 - lx1
        lcenter_y, lcenter_x = (ly1 + ly2) // 2, (lx1 + lx2) // 2
        lnew_h, lnew_w = int(lh * scale), int(lw * scale)
        lcrop_y1 = max(0, lcenter_y - lnew_h // 2)
        lcrop_y2 = min(h, lcenter_y + lnew_h // 2)
        lcrop_x1 = max(0, lcenter_x - lnew_w // 2)
        lcrop_x2 = min(w, lcenter_x + lnew_w // 2)
        
        # Extract and resize the eye region
        left_eye = img_array[lcrop_y1:lcrop_y2, lcrop_x1:lcrop_x2]
        left_eye_resized = np.array(Image.fromarray((left_eye * 255).astype(np.uint8)).resize((lx2-lx1, ly2-ly1)))
        left_eye_resized = left_eye_resized / 255.0
        
        # Place back in the image
        modified[ly1:ly2, lx1:lx2] = left_eye_resized
        
        # Right eye (similar process)
        ry1, ry2, rx1, rx2 = right_eye_region
        rh, rw = ry2 - ry1, rx2 - rx1
        rcenter_y, rcenter_x = (ry1 + ry2) // 2, (rx1 + rx2) // 2
        rnew_h, rnew_w = int(rh * scale), int(rw * scale)
        rcrop_y1 = max(0, rcenter_y - rnew_h // 2)
        rcrop_y2 = min(h, rcenter_y + rnew_h // 2)
        rcrop_x1 = max(0, rcenter_x - rnew_w // 2)
        rcrop_x2 = min(w, rcenter_x + rnew_w // 2)
        
        right_eye = img_array[rcrop_y1:rcrop_y2, rcrop_x1:rcrop_x2]
        right_eye_resized = np.array(Image.fromarray((right_eye * 255).astype(np.uint8)).resize((rx2-rx1, ry2-ry1)))
        right_eye_resized = right_eye_resized / 255.0
        
        modified[ry1:ry2, rx1:rx2] = right_eye_resized
        
    elif operation == "close":
        # Simulate eye closing by darkening and reducing contrast
        for y1, y2, x1, x2 in [left_eye_region, right_eye_region]:
            # Get eye region
            eye = modified[y1:y2, x1:x2]
            
            # Apply darkening proportional to strength
            eye_darkened = eye * (1 - strength * 0.5)
            
            # Reduce contrast
            eye_mean = np.mean(eye_darkened, axis=(0, 1), keepdims=True)
            eye_reduced_contrast = eye_mean + (eye_darkened - eye_mean) * (1 - strength * 0.7)
            
            # Apply back to the image
            modified[y1:y2, x1:x2] = eye_reduced_contrast
            
    elif operation == "color":
        # Change eye color (simplistic approach - blue tint)
        blue_tint = np.array([0.0, 0.0, strength])
        
        for y1, y2, x1, x2 in [left_eye_region, right_eye_region]:
            # Get eye region
            eye = modified[y1:y2, x1:x2]
            
            # Find "eye-like" pixels (usually darker than surroundings)
            eye_mask = np.mean(eye, axis=2) < np.mean(eye)
            eye_mask = eye_mask[:, :, np.newaxis]
            
            # Apply blue tint
            modified[y1:y2, x1:x2] = eye * (1 - eye_mask * strength) + blue_tint * eye_mask * strength
            
    elif operation == "move":
        # Move eyes slightly apart or together
        move_distance = int(strength * eye_w * 0.6)  # How far to move
        
        if move_distance > 0:
            # Moving eyes apart - shift left eye left and right eye right
            # Left eye - move left
            ly1, ly2, lx1, lx2 = left_eye_region
            new_lx1, new_lx2 = max(0, lx1 - move_distance), max(move_distance, lx2 - move_distance)
            
            # Copy eye region to new position
            if new_lx1 < lx1:
                # Fill gap with skin tone (simple approach - use nearby pixels)
                skin_sample = img_array[ly1:ly2, lx2:min(w, lx2+10)]
                skin_color = np.mean(skin_sample, axis=(0, 1))
                modified[ly1:ly2, lx1-move_distance:lx1] = skin_color
            
            # Copy eye to new position
            width_to_copy = min(lx2 - lx1, new_lx2)
            modified[ly1:ly2, new_lx1:new_lx1+width_to_copy] = img_array[ly1:ly2, lx1:lx1+width_to_copy]
            
            # Right eye - move right
            ry1, ry2, rx1, rx2 = right_eye_region
            new_rx1, new_rx2 = min(w-move_distance, rx1 + move_distance), min(w, rx2 + move_distance)
            
            # Fill gap with skin tone
            if new_rx2 > rx2:
                skin_sample = img_array[ry1:ry2, max(0, rx1-10):rx1]
                skin_color = np.mean(skin_sample, axis=(0, 1))
                modified[ry1:ry2, rx2:new_rx2] = skin_color
                
            # Copy eye to new position
            width_to_copy = min(rx2 - rx1, w - new_rx1)
            modified[ry1:ry2, new_rx1:new_rx1+width_to_copy] = img_array[ry1:ry2, rx1:rx1+width_to_copy]
    
    return modified

def main():
    """Main function to demonstrate the eye manipulation"""
    # Download some sample faces
    face_paths = download_multiple_faces(start_idx=0, count=5)
    
    if not face_paths:
        print("Failed to download sample faces")
        return
    
    # Load the first face
    face_img = load_face_image(face_paths[0])
    
    # Show original
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 3, 1)
    plt.imshow(face_img)
    plt.axis('off')
    plt.title('Original')
    
    # Apply different eye manipulations
    operations = ["enlarge", "close", "color", "move"]
    strengths = [0.3, 0.7]
    
    for i, op in enumerate(operations):
        for j, strength in enumerate(strengths):
            modified = manipulate_eyes_simple(face_img, op, strength)
            plt.subplot(3, 3, 2 + i*2 + j)
            plt.imshow(modified)
            plt.axis('off')
            plt.title(f'{op.title()}, strength={strength}')
    
    plt.tight_layout()
    plt.show()
    
    print("Eye manipulation demonstration completed!")
    
if __name__ == "__main__":
    main()
