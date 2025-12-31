# Dataset Directory

This directory contains the ASAP (Automated Student Assessment Prize) dataset used for training and evaluation.

---

## 📥 Dataset Download

### ASAP Dataset

**Source:** Kaggle Competition - Automated Student Assessment Prize  
**URL:** https://www.kaggle.com/c/asap-aes/data

### Required Files

Download and place the following file in this directory:

```
data/
└── training_set_rel3.tsv
```

### Download Instructions

1. **Go to:** https://www.kaggle.com/c/asap-aes/data
2. **Sign in** to Kaggle (create free account if needed)
3. **Download:** `training_set_rel3.tsv`
4. **Place** the file in this `data/` directory

**File Size:** ~7.5 MB  
**Format:** Tab-separated values (TSV)  
**Encoding:** Latin-1

---

## 📊 Dataset Statistics

### Overview

| Characteristic | Value |
|---------------|-------|
| **Total Essays** | 12,938 |
| **Prompts** | 8 |
| **Score Range** | 0-3 to 0-60 (varies by prompt) |
| **Genres** | Argumentative, Narrative, Explanatory |
| **Language** | English |

### Essay Prompts

| Prompt | Type | Essay Set | Score Range | Essays |
|--------|------|-----------|-------------|--------|
| 1 | Persuasive/Narrative/Expository | 1 | 2-12 | 1,783 |
| 2 | Persuasive/Narrative/Expository | 2 | 1-6 | 1,800 |
| 3 | Source Dependent Responses | 3 | 0-3 | 1,726 |
| 4 | Source Dependent Responses | 4 | 0-3 | 1,772 |
| 5 | Source Dependent Responses | 5 | 0-4 | 1,805 |
| 6 | Source Dependent Responses | 6 | 0-4 | 1,800 |
| 7 | Persuasive/Narrative/Expository | 7 | 0-30 | 1,569 |
| 8 | Persuasive/Narrative/Expository | 8 | 0-60 | 723 |

### Prompt Details

**Prompt 1:** "More and more people use computers..."  
**Prompt 2:** "Censorship in libraries..."  
**Prompt 3:** "The Landlady" (creative/narrative)  
**Prompt 4:** "Winter hibiscus" (narrative)  
**Prompt 5:** "Narciso Rodriguez" (explanatory)  
**Prompt 6:** "The Mooring Mast" (source-based)  
**Prompt 7:** "Patience" (persuasive)  
**Prompt 8:** "Laughter" (source-based response)

---

## 🔧 Preprocessing

The dataset preprocessing is handled automatically by `aes_complete.py`:

### Automatic Processing

1. **Text Cleaning**
   - Remove special characters
   - Normalize whitespace
   - Handle encoding issues

2. **Tokenization**
   - BiLSTM: Word-level tokenization
   - BERT: WordPiece tokenization (bert-base-uncased)
   - RoBERTa: Byte-level BPE tokenization (roberta-base)

3. **Score Normalization**
   - Each prompt's scores normalized to [0, 1] range
   - Formula: `(score - min_score) / (max_score - min_score)`
   - Enables fair comparison across prompts

4. **Data Splitting**
   - Train: 70% (9,056 essays)
   - Validation: 15% (1,941 essays)
   - Test: 15% (1,941 essays)
   - Stratified by essay_set to maintain prompt distribution

---

## 📁 Directory Structure

After downloading the dataset:

```
data/
├── README.md                      # This file
└── training_set_rel3.tsv          # ASAP dataset (download required)
```

**Note:** The dataset file is not included in this repository due to size and licensing. Users must download it from Kaggle.

---

## 📄 File Format

### training_set_rel3.tsv

**Columns:**
- `essay_id` - Unique identifier
- `essay_set` - Prompt number (1-8)
- `essay` - Essay text
- `rater1_domain1` - First rater's score
- `rater2_domain1` - Second rater's score
- `domain1_score` - Resolved score (used for training)
- Additional domain scores for some prompts

**Example:**
```
essay_id    essay_set    essay    domain1_score
1           1            "Dear local..."    8
2           1            "Dear @CAPS1..."   9
```

---

## 🔍 Data Quality

### Inter-Rater Reliability

- **Average agreement:** Quadratic Weighted Kappa ~0.75-0.85
- **Two human raters** per essay
- **Adjudication process** for disagreements
- **Resolved scores** used as ground truth

### Known Issues

1. **Anonymization artifacts:** `@CAPS1`, `@ORGANIZATION1`, etc.
2. **Spelling/grammar errors:** Preserved from original student writing
3. **Variable length:** 20 to 600+ words per essay
4. **Prompt-specific scoring:** Different rubrics per prompt

---

## ⚖️ Licensing & Ethics

### Dataset License

- **Source:** Kaggle / Hewlett Foundation
- **Use:** Educational and research purposes
- **Attribution:** Required
- **Commercial use:** Subject to Kaggle terms

### Ethical Considerations

1. **Privacy:** All essays anonymized, no PII
2. **Informed consent:** Original collection involved consent
3. **Fair use:** Research and educational purposes only
4. **Bias awareness:** Dataset reflects demographics of original study

---

## 🔗 References

**Original Competition:**  
Hewlett Foundation. (2012). "Automated Student Assessment Prize." Kaggle.  
https://www.kaggle.com/c/asap-aes

**Dataset Paper:**  
Shermis, M. D., & Hamner, B. (2013). "Contrasting state-of-the-art automated scoring of essays." In *Handbook of automated essay evaluation* (pp. 313-346).

---

## 💡 Usage Tips

### For Training

```bash
# Ensure dataset is in correct location
ls data/training_set_rel3.tsv

# Train BERT model
python aes_complete.py --model bert --epochs 3 --batch_size 16

# Train RoBERTa model
python aes_complete.py --model roberta --epochs 3 --batch_size 16
```

### For Exploration

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/training_set_rel3.tsv', sep='\t', encoding='latin-1')

# View statistics
print(f"Total essays: {len(df)}")
print(f"Prompts: {df['essay_set'].nunique()}")
print(df['essay_set'].value_counts().sort_index())

# View sample essay
print(df.iloc[0]['essay'])
```

---

## ❓ FAQ

**Q: Why isn't the dataset included in the repository?**  
A: The dataset is 7.5MB and subject to Kaggle terms. Users must download it themselves to comply with licensing.

**Q: Can I use a different dataset?**  
A: Yes, but you'll need to modify the preprocessing code in `aes_complete.py` to match your dataset format.

**Q: What if I can't access Kaggle?**  
A: You need a free Kaggle account. Alternatively, search for ASAP dataset mirrors, but verify licensing.

**Q: How long does download take?**  
A: 1-2 minutes depending on internet speed (7.5MB file).

**Q: Is the dataset balanced?**  
A: Prompts have similar numbers of essays (700-1,800 each), though not perfectly balanced.

---

## 📞 Support

For dataset issues:
- **Kaggle:** Check competition discussion forum
- **Repository:** Open an issue on GitHub
- **Alternative:** Email Hewlett Foundation for dataset questions

---

**Last updated:** December 2025
