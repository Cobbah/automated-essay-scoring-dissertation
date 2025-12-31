@echo off
REM ============================================================================
REM AUTOMATED ESSAY SCORING - ENVIRONMENT SETUP SCRIPT
REM Quick setup for Windows
REM ============================================================================

echo ==========================================
echo AES Dissertation - Environment Setup
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies (this may take 10-15 minutes)...
pip install -r requirements.txt

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python -c "import pandas; print(f'✅ Pandas {pandas.__version__}')"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__}')"
python -c "import sklearn; print(f'✅ Scikit-learn {sklearn.__version__}')"

echo.
echo ==========================================
echo ✅ Setup complete!
echo ==========================================
echo.
echo To activate the environment in the future:
echo   venv\Scripts\activate
echo.
echo To start training:
echo   python aes_complete.py --train_csvs data\asap_training_fixed.csv --model_type bilstm --epochs 10 --batch_size 16 --save_dir results\bilstm_test
echo.
pause
