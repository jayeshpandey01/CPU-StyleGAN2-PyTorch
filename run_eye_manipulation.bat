@echo off
echo StyleGAN2 Eye Manipulation - CPU Version
echo =====================================
echo.

REM Check if TensorFlow weights are already converted
if exist stylegan2-ffhq-config-f_pt.pkl (
    echo Using existing converted PyTorch weights
) else (
    echo Converting TensorFlow weights to PyTorch format...
    python convert_tf_to_pt.py stylegan2-ffhq-config-f.pkl
    if %ERRORLEVEL% NEQ 0 (
        echo Error converting weights!
        pause
        exit /b 1
    )
)

REM Run diagnostic test
echo.
echo Running quick diagnostics...
python stylegan2_test_faces.py
if %ERRORLEVEL% NEQ 0 (
    echo Diagnostics failed!
    pause
    exit /b 1
)

REM Launch the notebook server
echo.
echo Launching Jupyter Notebook server...
jupyter notebook eye_manipulation_cpu_new.ipynb
