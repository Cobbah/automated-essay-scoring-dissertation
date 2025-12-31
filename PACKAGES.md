# Package Dependencies Documentation

## 📦 Complete Package List

This document explains every package used in this project and why it's needed.

---

## 🔥 Core Deep Learning (Essential)

### torch (1.12.1)
- **Purpose:** Main deep learning framework
- **Used for:** BiLSTM, BERT, RoBERTa model implementations
- **Size:** ~800 MB
- **Critical:** YES

### torchvision (0.13.1)
- **Purpose:** Computer vision utilities (dependency of torch)
- **Used for:** Image transformations (minimal use)
- **Size:** ~5 MB
- **Critical:** NO (but installed with torch)

### torchaudio (0.12.1)
- **Purpose:** Audio processing utilities (dependency of torch)
- **Used for:** Not directly used, but part of torch ecosystem
- **Size:** ~5 MB
- **Critical:** NO (but installed with torch)

---

## 🤗 Transformers & NLP (Essential for BERT/RoBERTa)

### transformers (4.21.3)
- **Purpose:** Pre-trained transformer models
- **Used for:** BERT and RoBERTa implementations
- **Size:** ~50 MB
- **Critical:** YES (for BERT/RoBERTa)

### tokenizers (0.12.1)
- **Purpose:** Fast tokenization for transformers
- **Used for:** Text preprocessing for BERT/RoBERTa
- **Size:** ~5 MB
- **Critical:** YES (for BERT/RoBERTa)

### sentencepiece (0.1.97)
- **Purpose:** Subword tokenization
- **Used for:** Some transformer tokenizers
- **Size:** ~2 MB
- **Critical:** YES (for some models)

### sacremoses (0.0.53)
- **Purpose:** Moses tokenizer/detokenizer
- **Used for:** Text normalization
- **Size:** ~1 MB
- **Critical:** NO (but helpful)

---

## 📊 Data Processing (Essential)

### pandas (1.4.3)
- **Purpose:** Data manipulation and analysis
- **Used for:** Loading CSV, data preprocessing, results storage
- **Size:** ~20 MB
- **Critical:** YES

### numpy (1.23.1)
- **Purpose:** Numerical computing
- **Used for:** Array operations, mathematical functions
- **Size:** ~15 MB
- **Critical:** YES

### scipy (1.9.0)
- **Purpose:** Scientific computing
- **Used for:** Statistical functions, correlation calculations
- **Size:** ~30 MB
- **Critical:** YES (for statistics)

---

## 📝 Natural Language Processing

### nltk (3.7)
- **Purpose:** Natural Language Toolkit
- **Used for:** Text preprocessing, tokenization
- **Size:** ~5 MB (+ ~200 MB for data)
- **Critical:** YES (for BiLSTM preprocessing)

### spacy (3.4.1)
- **Purpose:** Industrial-strength NLP
- **Used for:** Advanced text processing (optional)
- **Size:** ~50 MB (+ ~500 MB for models)
- **Critical:** NO (optional enhancement)

### gensim (4.2.0)
- **Purpose:** Topic modeling and word embeddings
- **Used for:** Loading GloVe embeddings for BiLSTM
- **Size:** ~30 MB (+ ~800 MB for GloVe)
- **Critical:** YES (for BiLSTM)

---

## 🎯 Machine Learning & Evaluation (Essential)

### scikit-learn (1.1.2)
- **Purpose:** Machine learning toolkit
- **Used for:** 
  - Quadratic Weighted Kappa (QWK)
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Train/test splitting
  - Data scaling
- **Size:** ~25 MB
- **Critical:** YES (for all evaluation metrics)

---

## ⚖️ Fairness & Explainability

### shap (0.41.0)
- **Purpose:** SHapley Additive exPlanations
- **Used for:** Model explainability analysis
- **Size:** ~10 MB
- **Critical:** NO (for explainability features only)
- **Note:** Framework implemented but not fully utilized due to time constraints

---

## 📈 Visualization

### matplotlib (3.5.3)
- **Purpose:** Plotting library
- **Used for:** Creating graphs, charts, visualizations
- **Size:** ~20 MB
- **Critical:** NO (for visualization only)

### seaborn (0.11.2)
- **Purpose:** Statistical data visualization
- **Used for:** Enhanced plots, heatmaps
- **Size:** ~5 MB
- **Critical:** NO (for visualization only)

### plotly (5.10.0)
- **Purpose:** Interactive plotting
- **Used for:** Interactive visualizations (optional)
- **Size:** ~30 MB
- **Critical:** NO (optional)

---

## 🔧 Utilities

### tqdm (4.64.0)
- **Purpose:** Progress bars
- **Used for:** Showing training progress, loop iterations
- **Size:** ~1 MB
- **Critical:** NO (but very helpful)

### requests (2.28.1)
- **Purpose:** HTTP library
- **Used for:** Downloading datasets, models
- **Size:** ~2 MB
- **Critical:** YES (for downloads)

### urllib3 (1.26.12)
- **Purpose:** HTTP client (dependency of requests)
- **Used for:** HTTP connections
- **Size:** ~1 MB
- **Critical:** YES (dependency)

---

## 📄 File Handling

### openpyxl (3.0.10)
- **Purpose:** Read/write Excel files (XLSX)
- **Used for:** Excel file support (optional)
- **Size:** ~5 MB
- **Critical:** NO (optional)

### xlrd (2.0.1)
- **Purpose:** Read Excel files (XLS)
- **Used for:** Legacy Excel support (optional)
- **Size:** ~1 MB
- **Critical:** NO (optional)

### python-docx (0.8.11)
- **Purpose:** Create/modify Word documents
- **Used for:** Document generation (optional)
- **Size:** ~2 MB
- **Critical:** NO (optional)

---

## 🔐 Configuration & Validation

### jsonschema (4.16.0)
- **Purpose:** JSON schema validation
- **Used for:** Validating configuration files
- **Size:** ~1 MB
- **Critical:** NO (optional)

### python-dotenv (0.20.0)
- **Purpose:** Environment variable management
- **Used for:** Configuration from .env files
- **Size:** <1 MB
- **Critical:** NO (optional)

---

## 📓 Development Tools (Optional)

### jupyter (1.0.0)
- **Purpose:** Interactive notebooks
- **Used for:** Data exploration, analysis
- **Size:** ~2 MB
- **Critical:** NO (development only)

### ipython (8.4.0)
- **Purpose:** Enhanced Python shell
- **Used for:** Interactive development
- **Size:** ~5 MB
- **Critical:** NO (development only)

### notebook (6.4.12)
- **Purpose:** Jupyter notebook interface
- **Used for:** Running notebooks
- **Size:** ~10 MB
- **Critical:** NO (development only)

---

## 🧪 Testing (Optional)

### pytest (7.1.3)
- **Purpose:** Testing framework
- **Used for:** Unit tests, integration tests
- **Size:** ~5 MB
- **Critical:** NO (development only)

### pytest-cov (3.0.0)
- **Purpose:** Code coverage for pytest
- **Used for:** Testing coverage reports
- **Size:** ~1 MB
- **Critical:** NO (development only)

---

## ✨ Code Quality (Optional)

### black (22.8.0)
- **Purpose:** Code formatter
- **Used for:** Automatic code formatting
- **Size:** ~2 MB
- **Critical:** NO (development only)

### flake8 (5.0.4)
- **Purpose:** Linting tool
- **Used for:** Code style checking
- **Size:** ~1 MB
- **Critical:** NO (development only)

### pylint (2.15.3)
- **Purpose:** Static code analysis
- **Used for:** Code quality checks
- **Size:** ~5 MB
- **Critical:** NO (development only)

---

## 📊 Summary Statistics

### Total Size Breakdown:
```
Essential packages:          ~1,000 MB
Optional packages:          ~150 MB
Downloaded models (first run): ~1,700 MB
Total initial download:      ~2,850 MB (~2.8 GB)
```

### Critical vs Optional:

**CRITICAL (must have):**
- torch, transformers, tokenizers
- pandas, numpy, scipy
- scikit-learn
- nltk, gensim
- requests

**OPTIONAL (nice to have):**
- matplotlib, seaborn, plotly (visualization)
- shap (explainability)
- jupyter, ipython (development)
- pytest, black, flake8 (testing/quality)
- openpyxl, xlrd (file formats)

---

## 🎯 Installation Strategies

### Minimal (Essential Only):
Use `requirements-minimal.txt` (~2 GB)
- Includes only critical packages
- Fastest installation
- Sufficient for training models

### Standard (Recommended):
Use `requirements.txt` (~2.8 GB)
- Includes all standard packages
- Full functionality
- Ready for research and analysis

### Development:
Use `requirements-dev.txt` (~3 GB)
- Includes everything
- Development tools
- Testing frameworks
- For contributors

---

## 📝 Version Pinning Rationale

All versions are pinned (exact versions specified) to ensure:
1. **Reproducibility:** Same versions = same results
2. **Stability:** Tested combinations work together
3. **Compatibility:** No breaking changes from updates
4. **Research integrity:** Results can be verified

---

## 🔄 Updating Packages

To update to latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

⚠️ **Warning:** May introduce breaking changes. Test thoroughly after updating.

For research reproducibility, keep versions as specified.

---

## 💡 Tips

1. **Use virtual environment:** Always isolate project dependencies
2. **Check versions:** After install, verify with `pip list`
3. **GPU users:** Install CUDA-specific PyTorch first
4. **Storage:** Ensure 10GB+ free space before installing
5. **First run:** Expect additional downloads for model weights

---

## 📞 Support

Issues with packages? Check:
1. Python version (3.9+ required)
2. Pip version (latest recommended)
3. Virtual environment activated
4. Sufficient disk space
5. Internet connection for downloads

---

**Last Updated:** January 2026
**Project:** MSc Dissertation - Automated Essay Scoring
**Author:** Theophilus Kweku Cobbah
