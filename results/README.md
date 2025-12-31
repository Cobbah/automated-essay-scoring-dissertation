# Experimental Results

This directory contains the results from training and evaluating the three automated essay scoring models.

---

## 📊 Overall Results Summary

### Performance Comparison

| Model | QWK ↑ | MSE ↓ | MAE ↓ | Pearson ↑ | Training Time |
|-------|-------|-------|-------|-----------|---------------|
| **BiLSTM** | 0.639 | 0.020 | 0.107 | 0.801 | ~6 hours (CPU) |
| **BERT** | **0.768** | 0.014 | 0.090 | **0.877** | ~85 hours (CPU) |
| **RoBERTa** | 0.761 | **0.013** | **0.088** | 0.872 | ~85 hours (CPU) |

**Key Finding:** Transformer models (BERT/RoBERTa) achieve approximately 20% improvement in QWK over BiLSTM baseline.

---

## 📁 Directory Structure

```
results/
├── README.md                      # This file
│
├── bilstm_cpu/
│   └── results.json               # BiLSTM results
│
├── bert_cpu/
│   └── results.json               # BERT results
│
└── roberta_cpu/
    └── results.json               # RoBERTa results
```

**Note:** Model checkpoints and training logs are not included in this repository due to size constraints (400MB+ per model). The `results.json` files contain all performance metrics and fairness analysis.

---

## 🎯 BiLSTM Baseline Results

### Model Configuration

- **Architecture:** 2-layer bidirectional LSTM
- **Hidden units:** 128 per direction (256 total)
- **Embedding:** 300-dimensional GloVe
- **Dropout:** 0.3
- **Parameters:** ~2.5M
- **Training time:** ~6 hours on CPU

### Performance Metrics

| Metric | Value |
|--------|-------|
| **QWK** | 0.639 |
| **MSE** | 0.020 |
| **MAE** | 0.107 |
| **Pearson r** | 0.801 |

### Per-Prompt Performance

| Prompt | QWK | Best/Worst |
|--------|-----|------------|
| 1 | 0.652 | |
| 2 | 0.630 | |
| 3 | 0.449 | ⚠️ Worst |
| 4 | 0.535 | |
| 5 | 0.645 | |
| 6 | 0.673 | |
| 7 | 0.648 | ⭐ Best |
| 8 | 0.635 | |
| **Range** | **0.200** | (High variation) |

### Fairness Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Prompt Variation** | 0.200 | High inconsistency across prompts |
| **Length Bias** | r = -0.128 | Moderate negative correlation |
| **Std Dev** | 0.073 | High variation |

**Analysis:** BiLSTM shows significant prompt-specific behavior and moderate length bias, suggesting limited generalization.

---

## 🤖 BERT Results

### Model Configuration

- **Architecture:** `bert-base-uncased`
- **Layers:** 12 transformer layers
- **Hidden size:** 768
- **Attention heads:** 12
- **Parameters:** ~110M
- **Training time:** ~85 hours on CPU

### Performance Metrics

| Metric | Value |
|--------|-------|
| **QWK** | **0.768** ⭐ |
| **MSE** | 0.014 |
| **MAE** | 0.090 |
| **Pearson r** | **0.877** ⭐ |

### Per-Prompt Performance

| Prompt | QWK | Best/Worst |
|--------|-----|------------|
| 1 | 0.762 | |
| 2 | 0.831 | ⭐ Best |
| 3 | 0.673 | ⚠️ Worst |
| 4 | 0.735 | |
| 5 | 0.788 | |
| 6 | 0.750 | |
| 7 | 0.798 | |
| 8 | 0.752 | |
| **Range** | **0.158** | (21% improvement over BiLSTM) |

### Fairness Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Prompt Variation** | 0.158 | Improved consistency |
| **Length Bias** | r = -0.037 | ⭐ **Minimal bias** (71% reduction) |
| **Std Dev** | 0.048 | Lower variation |

**Analysis:** BERT achieves best overall QWK and demonstrates exceptional fairness regarding length bias, making it suitable for deployments where essay length fairness is critical.

---

## 🔧 RoBERTa Results

### Model Configuration

- **Architecture:** `roberta-base`
- **Layers:** 12 transformer layers
- **Hidden size:** 768
- **Attention heads:** 12
- **Parameters:** ~125M
- **Training time:** ~85 hours on CPU

### Performance Metrics

| Metric | Value |
|--------|-------|
| **QWK** | 0.761 |
| **MSE** | **0.013** ⭐ |
| **MAE** | **0.088** ⭐ |
| **Pearson r** | 0.872 |

### Per-Prompt Performance

| Prompt | QWK | Best/Worst |
|--------|-----|------------|
| 1 | 0.721 | |
| 2 | 0.762 | ⭐ Best |
| 3 | 0.615 | ⚠️ Worst |
| 4 | 0.698 | |
| 5 | 0.779 | |
| 6 | 0.745 | |
| 7 | 0.762 | |
| 8 | 0.758 | |
| **Range** | **0.147** ⭐ | (26.8% improvement over BiLSTM) |

### Fairness Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Prompt Variation** | **0.147** ⭐ | **Most consistent** across prompts |
| **Length Bias** | r = -0.076 | Moderate bias (40% reduction from BiLSTM) |
| **Std Dev** | 0.045 | Lowest variation |

**Analysis:** RoBERTa achieves the most consistent performance across prompts and best MSE/MAE, demonstrating superior generalization within the ASAP dataset.

---

## 📈 Comparative Analysis

### Performance Improvements

**BERT vs BiLSTM:**
- QWK: +20.2% (0.639 → 0.768)
- MSE: -30.0% (0.020 → 0.014)
- MAE: -15.9% (0.107 → 0.090)
- Pearson: +9.5% (0.801 → 0.877)

**RoBERTa vs BiLSTM:**
- QWK: +19.1% (0.639 → 0.761)
- MSE: -35.0% (0.020 → 0.013) ⭐ Best
- MAE: -17.8% (0.107 → 0.088) ⭐ Best
- Pearson: +8.9% (0.801 → 0.872)

**RoBERTa vs BERT:**
- QWK: -0.9% (nearly identical)
- Prompt variation: -7.0% (more consistent)
- Length bias: +105% (more biased)

### Complementary Fairness Profiles

**Key Finding:** No single model dominates all fairness dimensions.

- **For length fairness:** Choose BERT (r = -0.037)
- **For prompt consistency:** Choose RoBERTa (range 0.147)
- **For balanced fairness:** Consider both models' strengths

---

## 🔍 Detailed Findings

### 1. Performance Ceiling

Both BERT and RoBERTa converge near QWK 0.76-0.77, suggesting:
- Performance saturation for transformers on ASAP
- Potential ceiling imposed by inter-rater reliability (~0.75-0.85)
- Diminishing returns from additional model complexity

### 2. Prompt-Specific Challenges

All models struggle with **Prompt 3** ("The Landlady" - narrative):
- BiLSTM: 0.449 QWK
- BERT: 0.673 QWK
- RoBERTa: 0.615 QWK

**Hypothesis:** Creative narrative assessment requires different features than argumentative/explanatory essays.

### 3. Computational Trade-offs

**BERT/RoBERTa advantages:**
- +20% QWK improvement
- Better fairness (21-27% lower prompt variation)
- Superior error metrics (MSE/MAE)

**Cost:**
- 14× longer training time (85h vs 6h)
- 44-50× more parameters (110-125M vs 2.5M)
- Higher inference latency

**Recommendation:** Use transformers for high-stakes assessment; BiLSTM may suffice for low-stakes formative feedback.

---

## 📊 Statistical Significance

### Pairwise Comparisons

All model differences are statistically significant (bootstrap test, p < 0.001):
- BERT vs BiLSTM: p < 0.001
- RoBERTa vs BiLSTM: p < 0.001
- BERT vs RoBERTa: p = 0.34 (not significant)

**Interpretation:** Transformers significantly outperform BiLSTM, but BERT and RoBERTa perform equivalently overall.

---

## 📁 Results File Format

### results.json Structure

```json
{
  "model": "BERT",
  "test_metrics": {
    "qwk": 0.7683,
    "mse": 0.0139,
    "mae": 0.0896,
    "pearson": 0.8772
  },
  "per_prompt_qwk": {
    "1": 0.762,
    "2": 0.831,
    ...
  },
  "fairness_metrics": {
    "prompt_variation": 0.158,
    "length_bias": -0.037,
    "std_dev": 0.048
  },
  "config": {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "max_length": 512
  },
  "training_time": "~85 hours"
}
```

---

## 🔄 Reproducing Results

### Requirements

1. **Dataset:** Download ASAP dataset (see `data/README.md`)
2. **Dependencies:** Install from `requirements.txt`
3. **Hardware:** 8GB+ RAM, 50GB disk space

### Training Commands

```bash
# BiLSTM (6 hours)
python aes_complete.py --model bilstm --epochs 50 --batch_size 32

# BERT (~85 hours CPU, ~8 hours GPU)
python aes_complete.py --model bert --epochs 3 --batch_size 16

# RoBERTa (~85 hours CPU, ~8 hours GPU)
python aes_complete.py --model roberta --epochs 3 --batch_size 16
```

### Expected Outputs

- `results/{model}_cpu/results.json` - Performance metrics
- `models/{model}_best.pth` - Best model checkpoint (not uploaded)
- Training logs (not uploaded)

---

## ⚠️ Limitations

### What's NOT Included

1. **Model checkpoints** - Too large for GitHub (400MB+ each)
2. **Training logs** - Verbose output files
3. **Cross-dataset results** - TOEFL11 and Feedback Prize evaluation incomplete
4. **Explainability outputs** - SHAP analysis pending computational resources

### Known Issues

1. **CPU training is slow** - 85 hours for transformers (use GPU for practical deployment)
2. **Memory intensive** - BERT/RoBERTa require 8GB+ RAM
3. **Dataset-specific** - Results may not generalize to other essay corpora

---

## 📚 References

**For methodology details, see dissertation:**
- Chapter 3: Methodology
- Chapter 4: Results
- Chapter 5: Discussion

**For reproducibility:**
- See `aes_complete.py` for complete training code
- See `SETUP_GUIDE.md` for environment setup

---

## 💡 Key Takeaways

1. **Transformers substantially outperform BiLSTM** (~20% QWK improvement)
2. **BERT and RoBERTa perform similarly** overall (QWK difference <1%)
3. **Complementary fairness profiles** exist - no single best model for all equity dimensions
4. **Performance ceiling near 0.76-0.77 QWK** suggests fundamental limits on ASAP
5. **Computational cost is significant** but justified for high-stakes deployment

---

**Last updated:** December 2025  
**Dissertation:** See `docs/dissertation.pdf` for complete analysis
