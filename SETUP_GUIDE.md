# Installation & Setup Guide

## 📋 Prerequisites

- **Python 3.9+** (tested on 3.9.x and 3.10.x)
- **pip** (Python package manager)
- **Git** (for cloning repository)
- **8GB+ RAM** (16GB recommended for BERT/RoBERTa)
- **CUDA GPU** (optional, for faster training)

---

## 🚀 Quick Start (5 minutes)

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/automated-essay-scoring-dissertation.git
cd automated-essay-scoring-dissertation
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option A - Full Installation (Recommended):**
```bash
pip install -r requirements.txt
```

**Option B - Minimal Installation (Faster):**
```bash
pip install -r requirements-minimal.txt
```

### Step 4: Download Dataset

1. Download ASAP dataset from [Kaggle](https://www.kaggle.com/c/asap-aes/data)
2. Place `training_set_rel3.tsv` in `data/` folder
3. Or let the script download it automatically (if available)

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

✅ You're ready to train models!

---

## 🎯 Training Models

### BiLSTM Baseline (~6 hours on CPU)

```bash
python aes_complete.py \
    --train_csvs "data/asap_training_fixed.csv" \
    --model_type bilstm \
    --epochs 10 \
    --batch_size 16 \
    --save_dir results/bilstm_cpu \
    --do_fairness
```

### BERT Transformer (~85 hours on CPU, ~6 hours on GPU)

```bash
python aes_complete.py \
    --train_csvs "data/asap_training_fixed.csv" \
    --model_type transformer \
    --model_name bert-base-uncased \
    --epochs 3 \
    --batch_size 4 \
    --save_dir results/bert_cpu \
    --do_fairness
```

### RoBERTa Transformer (~85 hours on CPU, ~6 hours on GPU)

```bash
python aes_complete.py \
    --train_csvs "data/asap_training_fixed.csv" \
    --model_type transformer \
    --model_name roberta-base \
    --epochs 3 \
    --batch_size 4 \
    --save_dir results/roberta_cpu \
    --do_fairness
```

---

## 🔍 Checking Results

### Single Model Results

```bash
python check_results.py results/bert_cpu
```

### Compare Multiple Models

```bash
python check_results.py results/bilstm_cpu results/bert_cpu results/roberta_cpu
```

---

## 💻 GPU Setup (Optional but Recommended)

### Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### Install PyTorch with CUDA Support

**For CUDA 11.6:**
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

**For CUDA 11.7:**
```bash
pip install torch==1.12.1+cu117 torchvision==0.13.1+cu117 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

**For CUDA 11.8:**
```bash
pip install torch==1.12.1+cu118 torchvision==0.13.1+cu118 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Package Descriptions

### Core Packages

- **torch (1.12.1)** - Deep learning framework
- **transformers (4.21.3)** - Hugging Face transformers (BERT, RoBERTa)
- **pandas (1.4.3)** - Data manipulation
- **numpy (1.23.1)** - Numerical computing
- **scikit-learn (1.1.2)** - Machine learning utilities & metrics

### NLP Packages

- **gensim (4.2.0)** - Word2Vec, GloVe embeddings for BiLSTM
- **nltk (3.7)** - Natural language toolkit
- **spacy (3.4.1)** - Advanced NLP

### Evaluation & Analysis

- **scipy (1.9.0)** - Statistical functions
- **shap (0.41.0)** - Model explainability

### Visualization

- **matplotlib (3.5.3)** - Plotting
- **seaborn (0.11.2)** - Statistical visualization

### Utilities

- **tqdm (4.64.0)** - Progress bars
- **openpyxl (3.0.10)** - Excel file handling

---

## 🐛 Troubleshooting

### Problem: "No module named 'torch'"

**Solution:**
```bash
pip install torch transformers
```

### Problem: "CUDA out of memory"

**Solution:** Reduce batch size
```bash
python aes_complete.py --batch_size 2  # instead of 4
```

### Problem: "Can't download dataset"

**Solution:** Manually download from Kaggle and place in `data/` folder

### Problem: PyTorch version mismatch

**Solution:** Reinstall PyTorch
```bash
pip uninstall torch torchvision torchaudio
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

### Problem: ImportError for transformers

**Solution:**
```bash
pip install --upgrade transformers tokenizers
```

---

## 📊 System Requirements

### Minimum (CPU Training)

- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB free
- Training time: BiLSTM ~6h, BERT/RoBERTa ~85h

### Recommended (GPU Training)

- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA with 8GB+ VRAM (GTX 1080, RTX 2060, or better)
- Storage: 10GB free
- Training time: BiLSTM ~30min, BERT/RoBERTa ~6h

### Optimal (Fast GPU Training)

- CPU: 16+ cores
- RAM: 32GB
- GPU: NVIDIA with 16GB+ VRAM (RTX 3090, A100)
- Storage: 20GB free (SSD)
- Training time: BiLSTM ~15min, BERT/RoBERTa ~2-3h

---

## 🔄 Updating Dependencies

To update all packages to latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

To update specific package:

```bash
pip install --upgrade transformers
```

---

## 📝 Notes

- First run will download pre-trained models (~500MB for BERT, ~500MB for RoBERTa)
- GloVe embeddings (~800MB) will download on first BiLSTM training
- All downloads are cached for subsequent runs
- Results are saved automatically in JSON format
- Models can be interrupted and resumed (checkpoints saved)

---

## ✅ Verification Checklist

Before training, verify:

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] All packages installed successfully
- [ ] PyTorch can import without errors
- [ ] Transformers can import without errors
- [ ] Dataset file present in `data/` folder
- [ ] At least 10GB free disk space

Run verification:
```bash
python -c "import torch, transformers, pandas, numpy, sklearn, gensim; print('✅ All packages OK!')"
```

---

## 🎓 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Check Python version compatibility
5. Open an issue on GitHub if problem persists

---

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [ASAP Dataset Info](https://www.kaggle.com/c/asap-aes)
- [Dissertation PDF](docs/dissertation.pdf)

---

**Ready to start? Run your first training!** 🚀

```bash
python aes_complete.py --train_csvs "data/asap_training_fixed.csv" --model_type bilstm --epochs 10 --batch_size 16 --save_dir results/bilstm_test
```
