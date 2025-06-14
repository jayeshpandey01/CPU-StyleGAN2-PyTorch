# StyleGAN2 Eye Manipulation on CPU

**Extended implementation of StyleGAN2 for eye feature manipulation on CPU**  
_Based on [StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)_

---

## Key Features

- CPU-compatible StyleGAN2 implementation
- Eye feature manipulation (size, openness, distance, color)
- Pre-computed eye direction vectors
- Robust tensor dimension handling
- Direct image processing alternative

---

## Quick Start

```powershell
# Run the simplified notebook
.\run_eye_manipulation_simple.bat

# OR run the direct image processing script (fallback)
.\run_simple.bat
```

---

## User Guide

For a complete step-by-step guide, see the [USER_GUIDE.md](USER_GUIDE.md). This guide provides detailed instructions for both beginners and advanced users.

---

## Comprehensive Documentation

1. [**README_CPU_EYE_MANIPULATION.md**](README_CPU_EYE_MANIPULATION.md): Complete project overview  
2. [**TECHNICAL_DETAILS.md**](TECHNICAL_DETAILS.md): Technical implementation details  
3. [**PROCESS_DOCUMENTATION.md**](PROCESS_DOCUMENTATION.md): Step-by-step process documentation  
4. [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md): Solutions for common issues  

---

## Requirements

Tested on:

- PyTorch 1.3.1
- CUDA 10.1/10.2

---

## Usage

### 1. Prepare LMDB Datasets

```bash
python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH
```

This converts images to JPEG and pre-resizes them. You can create multi-resolution datasets with comma-separated `--size` arguments.

---

### 2. Training (Distributed)

```bash
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH
```

- Supports Weights & Biases logging via `--wandb`.

---

### 3. SWAGAN Support

This implementation supports [SWAGAN](https://arxiv.org/abs/2102.06108):

```bash
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --arch swagan --batch BATCH_SIZE LMDB_PATH
```

---

### 4. Convert Weights from Official Checkpoints

Clone the official [StyleGAN2 repo](https://github.com/NVlabs/stylegan2) and download `.pkl` checkpoints.

Convert weights:

```bash
python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl
```

---

### 5. Generate Samples

```bash
python generate.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT
```

Set `--size` if you trained at a different resolution.

---

### 6. Image Projection

```bash
python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...
```

---

### 7. Closed-Form Factorization

See [Closed-Form Factorization](https://arxiv.org/abs/2007.06600):

**Extract eigenvectors:**
```bash
python closed_form_factorization.py [CHECKPOINT]
```

**Apply extracted directions:**
```bash
python apply_factor.py -i [INDEX_OF_EIGENVECTOR] -d [DEGREE_OF_MOVE] -n [NUMBER_OF_SAMPLES] --ckpt [CHECKPOINT] [FACTOR_FILE]
```
Example:
```bash
python apply_factor.py -i 19 -d 5 -n 10 --ckpt [CHECKPOINT] factor.pt
```
Will generate 10 random samples, with latents moved along the 19th eigenvector with degree ±5.

![Sample of closed form factorization](doc/factor_index-13_degree-5.0.png)

---

## Implementation Process

This CPU-compatible eye manipulation implementation involved:

1. **CPU Operation Implementation**
    - Replacements for CUDA-specific operations
    - Simplified `upfirdn2d` and `fused_leaky_relu`
    - CPU-friendly noise injection

2. **Dimension Handling Fixes**
    - Fixed tensor dimension issues in latent vectors
    - Robust shape checking and validation
    - Automatic shape correction

3. **Pre-Computed Eye Directions**
    - Direction vectors for eye size, openness, distance, color
    - Support for combined feature manipulation
    - Normalized vectors for consistency

4. **Alternative Solutions**
    - Direct image processing for eye manipulation
    - Automated model downloading
    - Robust error handling and diagnostics

---

## Key Files and Components

- **Core Files**:
    - `eye_manipulation_simple.ipynb`: Main notebook for StyleGAN2-based eye manipulation
    - `simple_eye_manipulation.py`: Direct image processing alternative
    - `eye_directions.py`: Pre-computed eye feature direction vectors
    - `download_models.py`: Download official StyleGAN2 weights
    - `op/cpu_ops.py`: CPU-compatible operations for StyleGAN2

- **Support Files**:
    - `TROUBLESHOOTING.md`: Guide for fixing common issues
    - `setup_stylegan2_ada.py`: StyleGAN2-ADA integration setup utility
    - `run_eye_manipulation_simple.bat`: Run the simplified notebook (Windows)
    - `run_simple.bat`: Run the direct image processing approach (Windows)

---

## Original StyleGAN2 Samples

![Sample with truncation](doc/sample.png)  
Sample from FFHQ (trained on 3.52M images)

![MetFaces sample with non-leaking augmentations](doc/sample-metfaces.png)  
Sample from MetFaces with Non-leaking augmentations (150,000 iterations, trained on 4.8M images)

### Samples from Converted Weights

![Sample from FFHQ](doc/stylegan2-ffhq-config-f.png)  
Sample from FFHQ (1024px)

![Sample from LSUN Church](doc/stylegan2-church-config-f.png)  
Sample from LSUN Church (256px)

---

## License

- Model details and custom CUDA kernel codes from [NVIDIA StyleGAN2](https://github.com/NVlabs/stylegan2)
- LPIPS code from [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
- FID Inception V3 from [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
