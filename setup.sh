#!/bin/bash
# ============================================================================
# AUTOMATED ESSAY SCORING - ENVIRONMENT SETUP SCRIPT
# Quick setup for Unix/Linux/Mac
# ============================================================================

echo "=========================================="
echo "AES Dissertation - Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies (this may take 10-15 minutes)..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python -c "import pandas; print(f'✅ Pandas {pandas.__version__}')"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__}')"
python -c "import sklearn; print(f'✅ Scikit-learn {sklearn.__version__}')"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To start training:"
echo "  python aes_complete.py --train_csvs data/asap_training_fixed.csv --model_type bilstm --epochs 10 --batch_size 16 --save_dir results/bilstm_test"
echo ""
