# User Guide: StyleGAN2 Eye Manipulation on CPU

This guide provides step-by-step instructions for manipulating eye features in human faces using StyleGAN2 on CPU.

## Quick Start Options

You have two main options for eye manipulation:

### Option 1: StyleGAN2-Based Approach (Recommended)

This approach uses the full StyleGAN2 model to generate and manipulate high-quality face images.

1. **Run the batch file**:
   ```
   .\run_eye_manipulation_simple.bat
   ```

2. **Or open the notebook directly**:
   ```
   jupyter notebook eye_manipulation_simple.ipynb
   ```

3. **Run the cells in order** to:
   - Load the StyleGAN2 model
   - Generate a base face
   - Manipulate eye features

### Option 2: Direct Image Processing (Fallback)

If Option 1 doesn't work, this approach uses basic image processing on sample face images.

1. **Run the batch file**:
   ```
   .\run_simple.bat
   ```

2. **Or run the Python script directly**:
   ```
   python simple_eye_manipulation.py
   ```

## Eye Manipulation Features

You can manipulate these eye features:

1. **Eye Size**: Make the eyes larger or smaller
2. **Eye Openness**: Make the eyes more open or more closed
3. **Eye Distance**: Move the eyes further apart or closer together
4. **Eye Color**: Adjust the eye color brightness

## Using the Interactive Controls

In the simplified notebook, you'll find interactive controls:

```python
interact(
    interactive_eye_manipulation,
    size=widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0.0),
    openness=widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0.0),
    distance=widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0.0),
    color=widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0.0)
)
```

Adjust the sliders to see real-time changes in the face image.

## Tips for Best Results

1. **Start with Low Strength Values**: 
   - Begin with small values (-0.3 to 0.3)
   - Gradually adjust to avoid extreme distortions

2. **Try Different Base Faces**:
   - Change the random seed for variety
   - Some faces respond better to manipulation than others

3. **Combine Feature Adjustments**:
   - Combine eye size and openness for natural results
   - Adjust distance slightly for subtle changes

4. **Save Your Results**:
   - Use `plt.savefig("my_result.png")` to save images
   - Or add this code to save images:
   ```python
   from PIL import Image
   img_np = (img[0].cpu().permute(1, 2, 0).numpy() + 1) * 0.5
   img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
   Image.fromarray(img_np).save("generated_face.png")
   ```

## Troubleshooting

If you encounter issues:

1. **Pink/Magenta Noise Images**:
   - Try the direct image processing approach
   - Check that the model weights downloaded correctly

2. **Out of Memory Errors**:
   - Reduce the image size in the code (e.g., from 128 to 64)
   - Close other applications

3. **Slow Performance**:
   - StyleGAN2 is computationally intensive on CPU
   - Be patient or try the direct image processing approach

4. **Missing Dependencies**:
   - The batch files should install required packages
   - If not, manually install: `pip install torch matplotlib numpy pillow requests`

## Advanced Usage

For more advanced manipulation:

```python
# Import utilities
from eye_directions import get_combined_eye_direction

# Create a custom direction
custom_dir = get_combined_eye_direction(
    size=0.5,      # Larger eyes
    openness=0.3,  # Slightly more open
    distance=-0.2, # Slightly closer together
    color=0.1      # Slightly lighter color
)

# Apply the custom direction
img = manipulate_eyes_generic(
    g_ema,
    base_latent,
    custom_dir,
    strength=1.0
)
```

For more details, see the comprehensive documentation files in the repository.
