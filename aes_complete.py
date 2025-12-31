"""
Complete AES System - Verified Working Version
"""
import os, json, argparse, zipfile, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

TEXT_COLS = ["full_text", "text", "essay", "essay_text"]
SCORE_COLS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_scores(s):
    return pd.Series([0.5]*len(s)) if s.nunique() == 1 else (s - s.min()) / (s.max() - s.min() + 1e-12)

def detect_dataset_type(df):
    """Detect which dataset format we're dealing with."""
    cols = set(df.columns)
    
    # Check for ASAP dataset
    if "domain1_score" in cols or "essay_set" in cols:
        return "asap"
    
    # Check for TOEFL11 dataset
    elif "score" in cols and ("L1" in cols or "native_lang" in cols):
        return "toefl11"
    
    # Check for Feedback Prize dataset
    elif any(c in cols for c in SCORE_COLS):
        return "feedback_prize"
    
    # Check for generic/custom dataset with text and score columns
    elif any(text_col in cols for text_col in TEXT_COLS) and "score" in cols:
        return "custom"
    
    raise ValueError(f"Unknown dataset: {list(df.columns)}")

def load_dataset_csv(path, score_cols=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zip":
        with zipfile.ZipFile(path) as z:
            csvs = [n for n in z.namelist() if n.endswith(".csv")]
            df = pd.read_csv(z.open(next((n for n in csvs if "train" in n.lower()), csvs[0])))
    else:
        df = pd.read_csv(path)
    
    dtype = detect_dataset_type(df)
    text_col = next((c for c in TEXT_COLS if c in df.columns), None)
    if not text_col: raise ValueError(f"No text column found in {list(df.columns)}")
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].apply(clean_text)
    df["essay_length"] = df["text"].str.split().str.len()
    
    if dtype == "asap":
        score_cols = ["domain1_score"] if "domain1_score" in df.columns else ["score"]
        if "essay_set" in df.columns: 
            df["prompt_id"] = df["essay_set"]
        elif "assignment" in df.columns: 
            df["prompt_id"] = df["assignment"]
    
    elif dtype == "toefl11":
        score_cols = ["score"]
        df["native_lang"] = df.get("L1", "unknown")
    
    elif dtype == "feedback_prize":
        score_cols = score_cols or [c for c in SCORE_COLS if c in df.columns]
    
    elif dtype == "custom":
        # Handle custom dataset format
        score_cols = score_cols or ["score"]
        # Ensure all requested score columns exist
        score_cols = [c for c in score_cols if c in df.columns]
        if not score_cols:
            raise ValueError(f"No valid score columns found in dataset")
        # Keep prompt_id if it exists
        if "prompt_id" not in df.columns and "essay_set" in df.columns:
            df["prompt_id"] = df["essay_set"]
    
    # Normalize scores
    for col in score_cols:
        if col in df.columns: 
            df[f"norm_{col}"] = normalize_scores(df[col])
    
    df["source"] = os.path.basename(path)
    print(f"Loaded {len(df)} essays from {path} ({dtype}), scores: {score_cols}")
    return df, score_cols, dtype

def load_multiple_datasets(paths, score_cols=None):
    dfs, all_cols = [], []
    for path in paths:
        df, cols, _ = load_dataset_csv(path, score_cols)
        dfs.append(df)
        all_cols.extend(cols)
    return pd.concat(dfs, ignore_index=True), score_cols or list(set(all_cols))

def tokenize_texts(texts, tokenizer, max_len=512):
    enc = tokenizer(texts.tolist(), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

def build_vocab(texts, max_size=20000):
    counts = Counter()
    for text in texts: counts.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counts.most_common(max_size - 2): vocab[word] = len(vocab)
    return vocab

def texts_to_sequences(texts, vocab, max_len=512):
    seqs = []
    for text in texts:
        seq = [vocab.get(tok, vocab["<UNK>"]) for tok in text.lower().split()]
        seq = seq[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(seq))
        seqs.append(seq)
    return torch.tensor(seqs, dtype=torch.long)

def make_dataloader(ids, mask, labels, batch_size=8, shuffle=True):
    return DataLoader(TensorDataset(ids, mask, labels), batch_size=batch_size, shuffle=shuffle)

class BiLSTMAES(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden=128, layers=2, dropout=0.3, num_scores=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, layers, bidirectional=True, dropout=dropout if layers > 1 else 0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 64)
        self.out = nn.Linear(64, num_scores)
    
    def forward(self, ids, mask=None):
        x = self.embed(ids)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.out(nn.functional.relu(self.fc(self.drop(h))))

class TransformerAES(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_scores=1, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, hidden // 2)
        self.out = nn.Linear(hidden // 2, num_scores)
    
    def forward(self, ids, mask):
        out = self.encoder(ids, mask, output_attentions=True)
        cls = self.drop(out.last_hidden_state[:, 0])
        return self.out(nn.functional.relu(self.fc(cls))), out.attentions

def compute_qwk(y_true, y_pred, max_score=6):
    yt = np.clip(np.round(y_true * max_score), 0, max_score).astype(int)
    yp = np.clip(np.round(y_pred * max_score), 0, max_score).astype(int)
    return cohen_kappa_score(yt, yp, weights="quadratic")

def compute_metrics(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    ytd = np.clip(np.round(yt * 6), 0, 6).astype(int)
    ypd = np.clip(np.round(yp * 6), 0, 6).astype(int)
    return {
        "qwk": float(compute_qwk(yt, yp)),
        "mse": float(mean_squared_error(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "pearson_r": float(pearsonr(yt, yp)[0]),
        "exact_agreement": float(np.mean(ytd == ypd)),
        "adjacent_agreement": float(np.mean(np.abs(ytd - ypd) <= 1))
    }

def train_model(model, train_loader, val_loader, device, score_cols, epochs=3, lr=2e-5, model_type="transformer"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=len(train_loader) * epochs, pct_start=0.1)
    scaler, loss_fn = GradScaler(), nn.MSELoss()
    best_qwk, best_state = -1, None
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for ids, mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            with autocast():
                preds = model(ids, mask)[0] if model_type == "transformer" else model(ids, mask)
                loss = loss_fn(preds, labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            train_losses.append(loss.item())
        
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                preds = model(ids, mask)[0] if model_type == "transformer" else model(ids, mask)
                preds_all.append(preds.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
        
        preds_all, labels_all = np.vstack(preds_all), np.vstack(labels_all)
        qwks = [compute_qwk(labels_all[:, i], preds_all[:, i]) for i in range(labels_all.shape[1])]
        avg_qwk = np.mean(qwks)
        
        print(f"Epoch {epoch+1}: Loss={np.mean(train_losses):.4f}, Val QWK={avg_qwk:.4f}")
        history.append({"epoch": epoch+1, "train_loss": np.mean(train_losses), "val_qwk": avg_qwk})
        
        if avg_qwk > best_qwk:
            best_qwk, best_state = avg_qwk, model.state_dict().copy()
            print(f"  New best QWK: {best_qwk:.4f}")
    
    if best_state: model.load_state_dict(best_state)
    return model, history

def evaluate_model(model, tokenizer, df, score_cols, device, model_type="transformer", vocab=None, max_len=512):
    model.eval()
    if model_type == "transformer":
        ids, mask = tokenize_texts(df["text"], tokenizer, max_len)
    else:
        ids = texts_to_sequences(df["text"], vocab, max_len)
        mask = (ids != 0).long()
    
    labels = torch.tensor(df[[f"norm_{c}" for c in score_cols]].values, dtype=torch.float32)
    loader = make_dataloader(ids, mask, labels, 8, False)
    
    preds_all, labels_all = [], []
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask = ids.to(device), mask.to(device)
            preds = model(ids, mask)[0] if model_type == "transformer" else model(ids, mask)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    
    preds_all, labels_all = np.vstack(preds_all), np.vstack(labels_all)
    results = {col: compute_metrics(labels_all[:, i], preds_all[:, i]) for i, col in enumerate(score_cols)}
    return results, preds_all, labels_all

def fairness_analysis(model, tokenizer, df, score_cols, device, group_col="prompt_id", model_type="transformer", vocab=None):
    print(f"\nFAIRNESS ANALYSIS: {group_col}")
    if group_col not in df.columns: return {}, {}
    
    results = {}
    for gval in sorted(df[group_col].unique()):
        gdf = df[df[group_col] == gval]
        if len(gdf) < 10: continue
        metrics, _, _ = evaluate_model(model, tokenizer, gdf, score_cols, device, model_type, vocab)
        results[str(gval)] = metrics
        print(f"{group_col}={gval}: QWK={metrics[score_cols[0]]['qwk']:.4f}")
    
    if len(results) > 1:
        stats = {}
        for col in score_cols:
            qwks = [results[g][col]['qwk'] for g in results]
            stats[col] = {"qwk_range": float(max(qwks) - min(qwks)), "qwk_std": float(np.std(qwks))}
        return results, stats
    return results, {}

def analyze_length_bias(model, tokenizer, df, score_cols, device, model_type="transformer", vocab=None):
    _, preds, labels = evaluate_model(model, tokenizer, df, score_cols, device, model_type, vocab)
    residuals = preds - labels
    lengths = df["text"].str.split().str.len().values
    results = {}
    for i, col in enumerate(score_cols):
        corr, pval = pearsonr(lengths, residuals[:, i])
        interp = "No bias"
        if abs(corr) > 0.3: interp = f"Moderate bias ({'over-rewarding' if corr > 0 else 'penalizing'} length)"
        results[col] = {"correlation": float(corr), "p_value": float(pval), "interpretation": interp}
        print(f"{col}: r={corr:.4f}, {interp}")
    return results, lengths, residuals

def plot_training(history_df, save_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history_df["epoch"], history_df["train_loss"], marker='o')
    ax[0].set_title("Training Loss")
    ax[1].plot(history_df["epoch"], history_df["val_qwk"], marker='o', color='green')
    ax[1].set_title("Validation QWK")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_metrics.png", dpi=150)
    plt.close()

def save_config(args, save_dir):
    config = {k: v for k, v in vars(args).items()}
    config["timestamp"] = pd.Timestamp.now().isoformat()
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

def generate_report(results, save_dir):
    with open(f"{save_dir}/report.txt", "w") as f:
        f.write("="*80 + "\nAUTOMATED ESSAY SCORING REPORT\n" + "="*80 + "\n\n")
        if "test_metrics" in results:
            f.write("TEST METRICS:\n")
            for col, m in results["test_metrics"].items():
                f.write(f"\n{col}: QWK={m['qwk']:.4f}, MSE={m['mse']:.4f}, MAE={m['mae']:.4f}\n")
        if "fairness_stats" in results and results["fairness_stats"]:
            f.write("\nFAIRNESS:\n")
            for col, s in results["fairness_stats"].items():
                f.write(f"{col}: QWK range={s['qwk_range']:.4f}\n")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csvs", nargs="+", required=True)
    p.add_argument("--test_csvs", nargs="*", default=[])
    p.add_argument("--model_type", default="transformer", choices=["transformer", "bilstm"])
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--do_fairness", action="store_true")
    p.add_argument("--save_dir", default="artifacts")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading datasets...")
    df, score_cols = load_multiple_datasets(args.train_csvs)
    train_df, temp = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    vocab = None
    if args.model_type == "transformer":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_ids, train_mask = tokenize_texts(train_df["text"], tokenizer, args.max_len)
        val_ids, val_mask = tokenize_texts(val_df["text"], tokenizer, args.max_len)
        model = TransformerAES(args.model_name, len(score_cols))
        print(f"Loaded tokenizer: {args.model_name}")
    else:
        vocab = build_vocab(train_df["text"])
        tokenizer = None
        train_ids = texts_to_sequences(train_df["text"], vocab, args.max_len)
        train_mask = (train_ids != 0).long()
        val_ids = texts_to_sequences(val_df["text"], vocab, args.max_len)
        val_mask = (val_ids != 0).long()
        model = BiLSTMAES(len(vocab), num_scores=len(score_cols))
        print(f"Built vocabulary: {len(vocab)} tokens")
    
    train_labels = torch.tensor(train_df[[f"norm_{c}" for c in score_cols]].values, dtype=torch.float32)
    val_labels = torch.tensor(val_df[[f"norm_{c}" for c in score_cols]].values, dtype=torch.float32)
    
    train_loader = make_dataloader(train_ids, train_mask, train_labels, args.batch_size)
    val_loader = make_dataloader(val_ids, val_mask, val_labels, args.batch_size, False)
    
    print(f"\nInitializing model...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n{'='*60}\nTRAINING\n{'='*60}")
    model, history = train_model(model, train_loader, val_loader, device, score_cols, args.epochs, args.lr, args.model_type)
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{args.save_dir}/training_log.csv", index=False)
    plot_training(history_df, args.save_dir)
    
    torch.save(model.state_dict(), f"{args.save_dir}/model.pt")
    if tokenizer: tokenizer.save_pretrained(args.save_dir)
    if vocab:
        with open(f"{args.save_dir}/vocab.json", "w") as f:
            json.dump(vocab, f)
    
    all_results = {}
    print("\nEvaluating on test set...")
    test_metrics, _, _ = evaluate_model(model, tokenizer, test_df, score_cols, device, args.model_type, vocab, args.max_len)
    all_results["test_metrics"] = test_metrics
    
    for col, m in test_metrics.items():
        print(f"{col}: QWK={m['qwk']:.4f}, MSE={m['mse']:.4f}, MAE={m['mae']:.4f}")
    
    if args.do_fairness and "prompt_id" in test_df.columns:
        _, fairness_stats = fairness_analysis(model, tokenizer, test_df, score_cols, device, "prompt_id", args.model_type, vocab)
        all_results["fairness_stats"] = fairness_stats
    
    print("\nLength bias analysis...")
    length_bias, _, _ = analyze_length_bias(model, tokenizer, test_df, score_cols, device, args.model_type, vocab)
    all_results["length_bias"] = length_bias
    
    with open(f"{args.save_dir}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    save_config(args, args.save_dir)
    generate_report(all_results, args.save_dir)
    
    print(f"\n{'='*60}\nAll outputs saved to {args.save_dir}\n{'='*60}")

if __name__ == "__main__":
    main()