import torch
import numpy as np

# Pre-computed eye manipulation directions based on StyleGAN2 FFHQ model
# These directions focus on eye changes while keeping other facial features consistent

# Direction for eye size (larger positive values = larger eyes)
EYE_SIZE_DIRECTION = np.array([
    0.0, 0.02, 0.1, 0.04, 0.08, 0.15, 0.2, 0.1, 0.02, 0.01,  # 0-9
    0.01, 0.05, 0.2, 0.15, 0.1, 0.05, 0.02, 0.0, 0.0, 0.0,   # 10-19
    0.3, 0.4, 0.5, 0.3, 0.2, 0.1, 0.15, 0.2, 0.25, 0.15,     # 20-29 (strongest eye effect)
    0.1, 0.05, 0.05, 0.1, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0,    # 30-39
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 40-49
])

# Direction for eye openness (positive = more open, negative = more closed)
EYE_OPENNESS_DIRECTION = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.05, 0.0, 0.0,      # 0-9
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 10-19
    0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05,      # 20-29 (strongest effect)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 30-39
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 40-49
])

# Direction for eye distance/horizontal position (positive = wider apart)
EYE_DISTANCE_DIRECTION = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 0-9 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.15, 0.1, 0.05,      # 10-19
    0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.25, 0.2, 0.1, 0.05,     # 20-29
    0.05, 0.1, 0.15, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,     # 30-39
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 40-49
])

# Direction for eye color (positive = lighter eyes, negative = darker)
EYE_COLOR_DIRECTION = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 0-9
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 10-19
    0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.25, 0.2, 0.15, 0.1,      # 20-29
    0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # 30-39
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 40-49
])

def get_eye_direction(direction_type="size", latent_dim=512, device="cpu"):
    """
    Get a pre-computed direction for eye manipulation
    
    Args:
        direction_type: Type of eye manipulation ('size', 'openness', 'distance', 'color')
        latent_dim: Dimension of the latent space (default: 512)
        device: Device to put the tensor on
        
    Returns:
        torch.Tensor: Direction vector for the specified manipulation
    """
    # Select the appropriate direction
    if direction_type == "size":
        values = EYE_SIZE_DIRECTION
    elif direction_type == "openness":
        values = EYE_OPENNESS_DIRECTION
    elif direction_type == "distance":
        values = EYE_DISTANCE_DIRECTION
    elif direction_type == "color":
        values = EYE_COLOR_DIRECTION
    else:
        raise ValueError(f"Unknown direction type: {direction_type}")
    
    # Create a zeroed direction vector
    direction = np.zeros(latent_dim)
    
    # Fill in the values we have
    direction[:min(len(values), latent_dim)] = values[:min(len(values), latent_dim)]
    
    # Convert to tensor and normalize
    direction_tensor = torch.from_numpy(direction).float().to(device)
    direction_tensor = direction_tensor / max(direction_tensor.norm(), 1e-8)
    
    return direction_tensor.unsqueeze(0)  # Return with batch dimension [1, latent_dim]

def get_combined_eye_direction(size=0.0, openness=0.0, distance=0.0, color=0.0, latent_dim=512, device="cpu"):
    """
    Get a combined direction vector for multiple eye attributes
    
    Args:
        size: Weight for eye size (-1.0 to 1.0)
        openness: Weight for eye openness (-1.0 to 1.0)
        distance: Weight for eye distance (-1.0 to 1.0)
        color: Weight for eye color (-1.0 to 1.0)
        latent_dim: Dimension of latent space
        device: Device to put tensor on
        
    Returns:
        torch.Tensor: Combined direction vector [1, latent_dim]
    """
    # Get individual directions
    size_dir = get_eye_direction("size", latent_dim, device) if size != 0 else 0
    openness_dir = get_eye_direction("openness", latent_dim, device) if openness != 0 else 0
    distance_dir = get_eye_direction("distance", latent_dim, device) if distance != 0 else 0
    color_dir = get_eye_direction("color", latent_dim, device) if color != 0 else 0
    
    # Combine directions with weights
    combined_dir = (
        size * size_dir + 
        openness * openness_dir + 
        distance * distance_dir + 
        color * color_dir
    )
    
    # Handle the case where all weights are zero
    if isinstance(combined_dir, int) and combined_dir == 0:
        return torch.zeros(1, latent_dim, device=device)
        
    return combined_dir
