# Automated Essay Scoring using Deep Learning and Transformer-Based Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> MSc Data Science Dissertation | University of Sunderland | 2025/2026

**A comprehensive evaluation of automated essay scoring systems comparing BiLSTM, BERT, and RoBERTa models with focus on performance, fairness, and explainability.**

---

## 📋 Overview

This repository contains the complete implementation and analysis for my MSc Data Science dissertation investigating automated essay scoring (AES) using deep learning approaches. The research compares three architectures across multiple dimensions:

- **BiLSTM** - Recurrent neural network baseline
- **BERT** - Transformer-based model  
- **RoBERTa** - Optimized transformer model

### Key Research Questions

1. How do transformer models (BERT/RoBERTa) compare to BiLSTM for essay scoring?
2. Do models generalize across different datasets and populations?
3. What fairness issues arise (prompt variation, length bias, demographic differences)?
4. Can we explain model decisions using SHAP and attention visualization?

---

## 🎯 Key Findings

### Performance Results

| Model | QWK ↑ | MSE ↓ | MAE ↓ | Training Time |
|-------|-------|-------|-------|---------------|
| **BiLSTM** | 0.639 | 0.020 | 0.107 | ~6 hours |
| **BERT** | **0.768** | 0.014 | 0.090 | ~85 hours |
| **RoBERTa** | 0.761 | **0.013** | **0.088** | ~85 hours |

**🔑 Main Finding:** Transformers achieve ~20% QWK improvement over BiLSTM baseline.

### Fairness Analysis

| Model | Prompt Variation ↓ | Length Bias (r) ↓ |
|-------|-------------------|-------------------|
| BiLSTM | 0.200 | -0.128 |
| BERT | 0.158 | **-0.037** ⭐ |
| **RoBERTa** | **0.147** ⭐ | -0.076 |

**🔑 Main Finding:** Models show complementary fairness profiles - RoBERTa achieves best prompt consistency (26.8% improvement), while BERT minimizes length bias (71% reduction).

### Novel Contributions

1. **Complementary Fairness Profiles**: No single model dominates all fairness dimensions
2. **Performance Ceiling**: Both transformers saturate near QWK 0.76-0.77
3. **Fairness-Performance Trade-offs**: Quantified across multiple equity dimensions
4. **Comprehensive Framework**: Reproducible evaluation protocol for AES fairness

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- 50GB disk space for models and datasets

### Installation

#### Option 1: Automatic Setup (Recommended)

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
setup.bat
```

#### Option 2: Manual Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/automated-essay-scoring-dissertation.git
cd automated-essay-scoring-dissertation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Option 3: Minimal Installation

For quick testing with essential packages only:

```bash
pip install -r requirements-minimal.txt
```

### Download Dataset

1. Download ASAP dataset from [Kaggle](https://www.kaggle.com/c/asap-aes/data)
2. Place `training_set_rel3.tsv` in `data/` directory

---

## 💻 Usage

### Training Models

#### BiLSTM Baseline

```bash
python aes_complete.py --model bilstm --epochs 50 --batch_size 32
```

Expected training time: ~6 hours on CPU

#### BERT Transformer

```bash
python aes_complete.py --model bert --epochs 3 --batch_size 16
```

Expected training time: ~85 hours on CPU, ~8 hours on GPU (V100)

#### RoBERTa Transformer

```bash
python aes_complete.py --model roberta --epochs 3 --batch_size 16
```

Expected training time: ~85 hours on CPU, ~8 hours on GPU (V100)

---

## 📊 Repository Structure

```
automated-essay-scoring-dissertation/
│
├── README.md                       # This file
├── requirements.txt                # Full dependencies
├── requirements-minimal.txt        # Essential packages only
├── requirements-dev.txt            # Development tools
├── requirements-streamlit.txt      # For demo app
│
├── setup.sh                        # Unix/Linux/Mac setup script
├── setup.bat                       # Windows setup script
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
│
├── aes_complete.py                 # Main training script
├── streamlit_app.py                # Interactive demo app
│
├── SETUP_GUIDE.md                  # Detailed installation guide
├── PACKAGES.md                     # Package documentation
│
├── data/                           # Dataset directory
│   └── README.md                   # Dataset instructions
│
├── results/                        # Experimental results
│   ├── README.md                   # Results summary
│   ├── bilstm_cpu/
│   ├── bert_cpu/
│   └── roberta_cpu/
│
└── docs/                           # Documentation
    ├── README.md                   # Docs overview
    └── dissertation.pdf            # Full dissertation (23,709 words)
```

---

## 🎨 Interactive Demo

**Coming Soon:** A Streamlit web application for interactive essay scoring!

**Features:**
- Upload or paste essays for scoring
- Select model (BiLSTM, BERT, RoBERTa)
- Real-time score predictions
- Visualize model comparisons
- Explore fairness metrics

*Demo will be deployed to Streamlit Cloud after dissertation submission.*

---

## 🎓 Academic Context

**Author:** Theophilus Kweku Cobbah  
**Institution:** University of Sunderland  
**Program:** MSc Data Science  
**Supervisor:** Dr. Sardar Jaf  
**Academic Year:** 2025/2026

### Dissertation Details

- **Title:** Automated Essay Scoring using Deep Learning and Transformer-Based Models: A Comparative Evaluation of Fairness, Explainability, and Generalisability
- **Word Count:** 23,709
- **Chapters:** 6 (Introduction, Literature Review, Methodology, Results, Discussion, Conclusion)
- **References:** 52 academic sources

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{cobbah2026aes,
  author = {Cobbah, Theophilus Kweku},
  title = {Automated Essay Scoring using Deep Learning and Transformer-Based Models: 
           A Comparative Evaluation of Fairness, Explainability, and Generalisability},
  school = {University of Sunderland},
  year = {2026},
  type = {MSc Dissertation},
  url = {https://github.com/YOUR_USERNAME/automated-essay-scoring-dissertation}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dr. Sardar Jaf** - Dissertation supervisor
- **University of Sunderland** - Research facilities and resources
- **Hewlett Foundation** - ASAP dataset creation
- **HuggingFace** - Transformers library and pre-trained models

---

**⭐ If you find this work useful, please consider starring the repository!**

---

*Last updated: December 2025*
