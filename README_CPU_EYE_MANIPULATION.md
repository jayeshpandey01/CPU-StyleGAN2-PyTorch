# StyleGAN2 Eye Manipulation on CPU

This repository contains a solution for eye feature manipulation in human faces using StyleGAN2 running on CPU-only environments. The project addresses common issues with StyleGAN2 implementations on CPUs and provides multiple approaches to successfully generate and manipulate face images with a focus on eye features.

## Project Overview

StyleGAN2 is typically optimized for GPU execution. This project provides a CPU-compatible implementation with specific focus on:

1. Loading pretrained StyleGAN2 models properly on CPU
2. Replacing GPU-specific operations with CPU equivalents
3. Fixing tensor dimension issues in latent space manipulation
4. Providing pre-computed directions for eye feature manipulation
5. Offering fallback solutions when StyleGAN2 doesn't work as expected

## Problem Addressed

When running StyleGAN2 on CPU, users often encounter:
- Random pink/magenta noise instead of proper face images
- Tensor dimension mismatches
- Missing CUDA operations
- Memory issues with full-size models

## Solutions Implemented

### 1. CPU-Compatible Operations

We've created CPU-friendly replacements for GPU-specific operations:

- `cpu_ops.py` provides CPU implementations of:
  - LeakyReLU with scale and bias
  - Simplified upfirdn2d for up/downsampling
  - Various other StyleGAN2 operations

### 2. Pre-Computed Eye Directions

Instead of calculating latent space directions on the fly (which can be unreliable):

- `eye_directions.py` provides pre-computed vectors for:
  - Eye size adjustment
  - Eye openness
  - Eye distance (spacing)
  - Eye color
  - Combined feature manipulation

### 3. Simplified Implementation

Two main approaches are provided:

1. **StyleGAN2-based approach** (`eye_manipulation_simple.ipynb`):
   - Uses pretrained StyleGAN2 models
   - Properly handles tensor dimensions
   - Includes robust model loading
   - Uses pre-computed directions

2. **Direct image processing approach** (`simple_eye_manipulation.py`):
   - Uses basic image processing techniques
   - Works without StyleGAN2
   - Downloads sample faces from FFHQ dataset
   - Applies transformations directly to images

### 4. Model Weight Utilities

- `download_models.py`: Automatically downloads official StyleGAN2 weights
- `setup_stylegan2_ada.py`: Sets up NVIDIA's StyleGAN2-ADA for better compatibility

## Implementation Steps

The implementation followed these key steps:

1. **Analysis**
   - Identified issues with tensor dimensions in the original implementation
   - Found that weights weren't loading correctly in CPU mode
   - Discovered that certain GPU operations needed CPU replacements

2. **CPU Operation Implementation**
   - Created simple CPU versions of CUDA operations
   - Implemented simplified upfirdn2d operation
   - Added safe noise injection mechanism
   - Created CPU-compatible ToRGB module

3. **Dimension Handling**
   - Fixed tensor dimension issues in `generate_face` and `manipulate_eyes` functions
   - Added robust error handling to detect and fix dimension mismatches
   - Created debugging utilities to trace tensor shapes

4. **Feature Direction Extraction**
   - Created pre-computed direction vectors for specific eye features
   - Added support for combinations of different eye manipulations

5. **Alternative Solutions**
   - Created a simple image processing approach as fallback
   - Added support for downloading sample faces for processing
   - Created detailed troubleshooting guide

## Repository Structure

- **Core Files**
  - `eye_manipulation_simple.ipynb`: Main notebook for StyleGAN2-based eye manipulation
  - `simple_eye_manipulation.py`: Direct image processing approach
  - `eye_directions.py`: Pre-computed eye feature direction vectors
  - `download_models.py`: Utility to download official StyleGAN2 weights
  - `op/cpu_ops.py`: CPU-compatible operations for StyleGAN2

- **Support Files**
  - `TROUBLESHOOTING.md`: Detailed guide for fixing common issues
  - `setup_stylegan2_ada.py`: Setup utility for StyleGAN2-ADA integration
  - `run_eye_manipulation_simple.bat`: Windows batch file to run the simplified notebook
  - `run_simple.bat`: Windows batch file to run the direct image processing approach

## Usage Instructions

### Option 1: StyleGAN2-based Eye Manipulation

1. Run the simplified notebook:
```powershell
jupyter notebook eye_manipulation_simple.ipynb
```

2. Or use the batch file (which installs dependencies automatically):
```powershell
.\run_eye_manipulation_simple.bat
```

### Option 2: Direct Image Processing

For a quick demonstration without StyleGAN2:
```powershell
.\run_simple.bat
```

### Option 3: Manual Integration

To integrate the eye manipulation functionality in your own code:

```python
# Import utilities
from eye_directions import get_eye_direction, get_combined_eye_direction

# Get a direction for eye size manipulation
eye_size_direction = get_eye_direction("size", latent_dim=512, device="cpu")

# Manipulate eye size in a latent vector
modified_latent = base_latent + eye_size_direction * strength

# Generate face with modified latent vector
img = generate_face(g_ema, modified_latent)
```

## Troubleshooting

If you're still seeing pink/magenta noise instead of faces:

1. Check the model weights:
```powershell
python download_models.py --model ffhq
```

2. Verify tensor dimensions in your code:
```python
print(f"Latent shape: {latent_w.shape}")
print(f"Mean latent shape: {mean_latent.shape}")
```

3. Try the direct image processing approach:
```powershell
python simple_eye_manipulation.py
```

For more detailed guidance, see `TROUBLESHOOTING.md`.

## Requirements

- Python 3.7+
- PyTorch 1.7+ (CPU version)
- matplotlib
- numpy
- Pillow
- requests

## Credits

This project builds upon the StyleGAN2 implementation by NVIDIA, with modifications for CPU compatibility and eye feature manipulation.

## License

This project follows the same license as the original StyleGAN2 by NVIDIA.
