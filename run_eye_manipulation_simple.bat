@echo off
echo Running StyleGAN2 Eye Manipulation (Simple CPU Version)
echo =====================================================

:: Check for Python and pip
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check for required packages
echo Checking required packages...
python -c "import torch" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

python -c "import matplotlib" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing matplotlib...
    pip install matplotlib
)

python -c "import requests" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing requests...
    pip install requests
)

python -c "import gdown" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing gdown...
    pip install gdown
)

python -c "import ipywidgets" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing ipywidgets...
    pip install ipywidgets
)

:: Check for the model file
if not exist stylegan2-ffhq-config-f.pkl (
    echo StyleGAN2 model not found. Downloading...
    python download_models.py --model ffhq
)

:: Run the notebook
echo Starting Jupyter notebook...
jupyter notebook eye_manipulation_simple.ipynb

echo Done.
pause
