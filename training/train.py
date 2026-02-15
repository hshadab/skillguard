#!/usr/bin/env python3
"""
SkillGuard classifier training pipeline.

Trains a 3-layer MLP (28 -> 32 -> 32 -> 4) on synthetic skill safety data,
quantizes weights to fixed-point i32 (scale=128), and exports Rust constants.

Usage:
    pip install -r requirements.txt
    python train.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIM = 28
HIDDEN_DIM = 32
OUTPUT_DIM = 4
SCALE = 128  # fixed-point scale factor (2^7)
EPOCHS = 200
LR = 0.001
BATCH_SIZE = 64
SEED = 42

CLASS_NAMES = ["SAFE", "CAUTION", "DANGEROUS", "MALICIOUS"]

FEATURE_NAMES = [
    "shell_exec_count",          # 0
    "network_call_count",        # 1
    "fs_write_count",            # 2
    "env_access_count",          # 3
    "credential_patterns",       # 4
    "external_download",         # 5
    "obfuscation_score",         # 6
    "privilege_escalation",      # 7
    "persistence_mechanisms",    # 8
    "data_exfiltration",         # 9
    "skill_md_line_count",       # 10
    "script_file_count",         # 11
    "dependency_count",          # 12
    "author_account_age",        # 13
    "author_skill_count",        # 14
    "stars",                     # 15
    "downloads",                 # 16
    "has_virustotal_report",     # 17
    "vt_malicious_flags",        # 18
    "password_protected_archives", # 19
    "reverse_shell_patterns",    # 20
    "llm_secret_exposure",       # 21
    "entropy_score",             # 22
    "non_ascii_ratio",           # 23
    "max_line_length",           # 24
    "comment_ratio",             # 25
    "domain_count",              # 26
    "string_obfuscation_score",  # 27
]


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(rng: np.random.Generator, n_total: int = 2000):
    """Generate labeled samples across 4 classes with realistic distributions."""

    n_safe = int(n_total * 0.30)       # ~600
    n_caution = int(n_total * 0.20)    # ~400
    n_dangerous = int(n_total * 0.25)  # ~500
    n_malicious = n_total - n_safe - n_caution - n_dangerous  # ~500

    samples = []
    labels = []

    # --- SAFE (~600): low risk signals, high reputation ---
    for _ in range(n_safe):
        f = np.zeros(INPUT_DIM, dtype=np.float32)
        f[0] = rng.integers(0, 3)     # shell_exec: very low
        f[1] = rng.integers(0, 10)    # network: low-moderate
        f[2] = rng.integers(0, 5)     # fs_write: low
        f[3] = rng.integers(0, 5)     # env_access: low
        f[4] = rng.integers(0, 3)     # credential: low
        f[5] = 0                       # no external download
        f[6] = 0                       # no obfuscation
        f[7] = 0                       # no priv esc
        f[8] = 0                       # no persistence
        f[9] = 0                       # no exfiltration
        f[10] = rng.integers(20, 128)  # good docs
        f[11] = rng.integers(0, 5)     # few scripts
        f[12] = rng.integers(0, 15)    # moderate deps
        f[13] = rng.integers(60, 128)  # established author
        f[14] = rng.integers(10, 100)  # many skills
        f[15] = rng.integers(30, 128)  # high stars
        f[16] = rng.integers(40, 128)  # high downloads
        f[17] = rng.choice([0, 128])   # VT report maybe
        f[18] = 0                      # no VT flags
        f[19] = 0                      # no password archives
        f[20] = 0                      # no reverse shell
        f[21] = 0                      # no LLM exposure
        f[22] = rng.uniform(3.0, 5.5) * SCALE / 8.0   # normal entropy
        f[23] = 0                      # no non-ascii
        f[24] = rng.integers(0, 50) * SCALE // 1000    # short lines
        f[25] = rng.uniform(0.1, 0.5) * SCALE          # moderate comments
        f[26] = rng.integers(0, 3) * SCALE // 20       # few domains
        f[27] = 0                      # no string obfuscation
        samples.append(f)
        labels.append(0)

    # --- CAUTION (~400): moderate signals, legitimate tool use ---
    for _ in range(n_caution):
        f = np.zeros(INPUT_DIM, dtype=np.float32)
        f[0] = rng.integers(2, 12)    # shell_exec: moderate
        f[1] = rng.integers(5, 30)    # network: moderate-high
        f[2] = rng.integers(2, 15)    # fs_write: moderate
        f[3] = rng.integers(3, 15)    # env_access: moderate
        f[4] = rng.integers(2, 8)     # credential: moderate
        f[5] = rng.choice([0, 128], p=[0.7, 0.3])  # sometimes downloads
        f[6] = rng.integers(0, 4)     # low obfuscation
        f[7] = rng.choice([0, 128], p=[0.8, 0.2])  # sometimes priv esc
        f[8] = rng.integers(0, 2)     # low persistence
        f[9] = rng.integers(0, 2)     # low exfiltration
        f[10] = rng.integers(10, 100) # moderate docs
        f[11] = rng.integers(1, 8)    # moderate scripts
        f[12] = rng.integers(3, 20)   # moderate deps
        f[13] = rng.integers(30, 110) # moderate-established author
        f[14] = rng.integers(3, 50)   # moderate skills
        f[15] = rng.integers(10, 80)  # moderate stars
        f[16] = rng.integers(10, 80)  # moderate downloads
        f[17] = rng.choice([0, 128])
        f[18] = rng.integers(0, 3) * SCALE // 20    # very few VT flags
        f[19] = 0
        f[20] = 0                     # no reverse shell
        f[21] = 0                     # no LLM exposure
        f[22] = rng.uniform(3.5, 5.5) * SCALE / 8.0
        f[23] = rng.uniform(0, 0.02) * SCALE / 0.5
        f[24] = rng.integers(30, 200) * SCALE // 1000
        f[25] = rng.uniform(0.05, 0.35) * SCALE
        f[26] = rng.integers(1, 8) * SCALE // 20
        f[27] = rng.integers(0, 2) * SCALE // 10
        samples.append(f)
        labels.append(1)

    # --- DANGEROUS (~500): credential/LLM exposure, low reputation ---
    for _ in range(n_dangerous):
        f = np.zeros(INPUT_DIM, dtype=np.float32)
        f[0] = rng.integers(3, 15)    # shell_exec: moderate-high
        f[1] = rng.integers(5, 35)    # network: moderate-high
        f[2] = rng.integers(2, 20)    # fs_write: moderate
        f[3] = rng.integers(8, 20)    # env_access: high
        f[4] = rng.integers(5, 10)    # credential: high
        f[5] = rng.choice([0, 128], p=[0.4, 0.6])
        f[6] = rng.integers(2, 10)    # moderate obfuscation
        f[7] = rng.choice([0, 128], p=[0.4, 0.6])
        f[8] = rng.integers(1, 4)     # some persistence
        f[9] = rng.integers(1, 4)     # some exfiltration
        f[10] = rng.integers(5, 60)   # sparse docs
        f[11] = rng.integers(1, 6)    # few scripts
        f[12] = rng.integers(2, 15)   # moderate deps
        f[13] = rng.integers(0, 60)   # new-ish author
        f[14] = rng.integers(0, 20)   # few skills
        f[15] = rng.integers(0, 30)   # low stars
        f[16] = rng.integers(0, 40)   # low downloads
        f[17] = rng.choice([0, 128], p=[0.6, 0.4])
        f[18] = rng.integers(1, 8) * SCALE // 20
        f[19] = rng.choice([0, 128], p=[0.8, 0.2])
        f[20] = rng.integers(0, 2)    # maybe reverse shell hints
        f[21] = rng.choice([0, 128], p=[0.3, 0.7])  # LLM exposure likely
        f[22] = rng.uniform(4.0, 6.5) * SCALE / 8.0
        f[23] = rng.uniform(0, 0.1) * SCALE / 0.5
        f[24] = rng.integers(40, 400) * SCALE // 1000
        f[25] = rng.uniform(0.0, 0.15) * SCALE
        f[26] = rng.integers(2, 12) * SCALE // 20
        f[27] = rng.integers(1, 5) * SCALE // 10
        samples.append(f)
        labels.append(2)

    # --- MALICIOUS (~500): reverse shells, obfuscation, high entropy ---
    for _ in range(n_malicious):
        f = np.zeros(INPUT_DIM, dtype=np.float32)
        f[0] = rng.integers(8, 20)    # shell_exec: high
        f[1] = rng.integers(5, 40)    # network: moderate-high
        f[2] = rng.integers(3, 25)    # fs_write: moderate-high
        f[3] = rng.integers(5, 20)    # env_access: high
        f[4] = rng.integers(3, 10)    # credential: moderate-high
        f[5] = rng.choice([0, 128], p=[0.2, 0.8])
        f[6] = rng.integers(5, 15)    # high obfuscation
        f[7] = rng.choice([0, 128], p=[0.3, 0.7])
        f[8] = rng.integers(2, 5)     # persistence
        f[9] = rng.integers(2, 5)     # exfiltration
        f[10] = rng.integers(0, 30)   # minimal docs
        f[11] = rng.integers(1, 8)    # several scripts
        f[12] = rng.integers(0, 10)   # few deps
        f[13] = rng.integers(0, 20)   # very new author
        f[14] = rng.integers(0, 5)    # very few skills
        f[15] = rng.integers(0, 10)   # no stars
        f[16] = rng.integers(0, 15)   # no downloads
        f[17] = rng.choice([0, 128], p=[0.5, 0.5])
        f[18] = rng.integers(3, 15) * SCALE // 20
        f[19] = rng.choice([0, 128], p=[0.5, 0.5])
        f[20] = rng.integers(2, 5) * SCALE // 5    # reverse shell patterns
        f[21] = rng.choice([0, 128], p=[0.4, 0.6])
        f[22] = rng.uniform(5.5, 8.0) * SCALE / 8.0  # high entropy
        f[23] = rng.uniform(0.05, 0.4) * SCALE / 0.5  # non-ascii
        f[24] = rng.integers(200, 1000) * SCALE // 1000  # long lines (minified)
        f[25] = rng.uniform(0.0, 0.05) * SCALE   # almost no comments
        f[26] = rng.integers(3, 20) * SCALE // 20
        f[27] = rng.integers(3, 10) * SCALE // 10  # heavy string obfuscation
        samples.append(f)
        labels.append(3)

    # --- Adversarial examples ---
    # Malicious with high stars (camouflaged)
    for _ in range(50):
        f = np.zeros(INPUT_DIM, dtype=np.float32)
        f[0] = rng.integers(8, 18)
        f[6] = rng.integers(6, 14)
        f[8] = rng.integers(2, 5)
        f[9] = rng.integers(2, 5)
        f[15] = rng.integers(60, 128)  # HIGH stars (camouflage)
        f[16] = rng.integers(50, 128)  # HIGH downloads
        f[20] = rng.integers(3, 5) * SCALE // 5
        f[22] = rng.uniform(6.0, 8.0) * SCALE / 8.0
        f[23] = rng.uniform(0.1, 0.35) * SCALE / 0.5
        f[24] = rng.integers(300, 900) * SCALE // 1000
        f[25] = rng.uniform(0.0, 0.03) * SCALE
        f[27] = rng.integers(4, 10) * SCALE // 10
        samples.append(f)
        labels.append(3)  # still MALICIOUS

    # Benign with high credential mentions (API/auth tools)
    for _ in range(50):
        f = np.zeros(INPUT_DIM, dtype=np.float32)
        f[1] = rng.integers(5, 25)
        f[3] = rng.integers(8, 18)
        f[4] = rng.integers(6, 10)    # HIGH credentials (legitimate auth tool)
        f[10] = rng.integers(30, 120)  # good docs
        f[13] = rng.integers(60, 128)
        f[14] = rng.integers(10, 80)
        f[15] = rng.integers(30, 100)
        f[16] = rng.integers(30, 100)
        f[22] = rng.uniform(3.5, 5.0) * SCALE / 8.0
        f[25] = rng.uniform(0.15, 0.4) * SCALE
        samples.append(f)
        labels.append(0)  # SAFE despite credentials

    # Evasion: minimal signals but suspicious new features
    for _ in range(50):
        f = np.zeros(INPUT_DIM, dtype=np.float32)
        f[0] = rng.integers(0, 3)     # low shell exec
        f[6] = rng.integers(0, 3)     # low obfuscation score
        # But new features reveal suspicion
        f[22] = rng.uniform(6.5, 8.0) * SCALE / 8.0  # very high entropy
        f[23] = rng.uniform(0.15, 0.4) * SCALE / 0.5  # non-ascii
        f[24] = rng.integers(500, 1000) * SCALE // 1000  # very long lines
        f[25] = 0                      # zero comments
        f[27] = rng.integers(5, 10) * SCALE // 10  # string obfuscation
        samples.append(f)
        labels.append(3)  # MALICIOUS (evasion attempt)

    X = np.array(samples, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Normalize features: apply same clip_scale as Rust
    # Features are already roughly in [0, 128] range from generation
    # Clip to [0, 128]
    X = np.clip(X, 0, SCALE)

    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SkillClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_val, y_val):
    model = SkillClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(torch.from_numpy(X_val))
                val_pred = val_out.argmax(dim=1).numpy()
                val_acc = (val_pred == y_val).mean()
                train_loss = total_loss / len(X_train)
                print(f"Epoch {epoch+1:3d} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_weights(model):
    """Multiply float weights by SCALE and round to i32."""
    state = model.state_dict()
    quantized = {}
    for name, tensor in state.items():
        q = torch.round(tensor * SCALE).to(torch.int32)
        quantized[name] = q.numpy()
    return quantized


# ---------------------------------------------------------------------------
# Rust export
# ---------------------------------------------------------------------------

def format_rust_array(name: str, data: np.ndarray, shape_comment: str,
                      neuron_comments: list[str] | None = None) -> str:
    """Format a weight array as a Rust const."""
    flat = data.flatten().tolist()
    lines = [f"const {name}: &[i32] = &["]

    if neuron_comments and len(data.shape) == 2:
        rows, cols = data.shape
        for i in range(rows):
            comment = neuron_comments[i] if i < len(neuron_comments) else f"Neuron {i}"
            row_vals = flat[i * cols : (i + 1) * cols]
            vals_str = ", ".join(str(v) for v in row_vals)
            lines.append(f"    // {comment}")
            lines.append(f"    {vals_str},")
    else:
        vals_str = ", ".join(str(v) for v in flat)
        lines.append(f"    {vals_str},")

    lines.append("];")
    return "\n".join(lines)


def export_rust_weights(quantized: dict, filepath: str):
    """Write quantized weights to a Rust file."""
    w1 = quantized["fc1.weight"]
    b1 = quantized["fc1.bias"]
    w2 = quantized["fc2.weight"]
    b2 = quantized["fc2.bias"]
    w3 = quantized["fc3.weight"]
    b3 = quantized["fc3.bias"]

    neuron_labels_w1 = [f"Neuron {i}" for i in range(HIDDEN_DIM)]
    neuron_labels_w2 = [f"Neuron {i}" for i in range(HIDDEN_DIM)]
    neuron_labels_w3 = [f"{CLASS_NAMES[i]} output" for i in range(OUTPUT_DIM)]

    content = f"""\
// Auto-generated by training/train.py — do not edit manually.
// Training seed: {SEED}
// Input features: {INPUT_DIM}, Hidden: {HIDDEN_DIM}, Output: {OUTPUT_DIM}
// Scale: {SCALE} (fixed-point)

/// Layer 1 weights: [{HIDDEN_DIM} hidden neurons, {INPUT_DIM} input features]
{format_rust_array("W1", w1, f"[{HIDDEN_DIM}, {INPUT_DIM}]", neuron_labels_w1)}

{format_rust_array("B1", b1.reshape(1, -1)[0:1].flatten()[np.newaxis, :].flatten(), f"[{HIDDEN_DIM}]")}

/// Layer 2 weights: [{HIDDEN_DIM} hidden neurons, {HIDDEN_DIM} neurons from layer 1]
{format_rust_array("W2", w2, f"[{HIDDEN_DIM}, {HIDDEN_DIM}]", neuron_labels_w2)}

{format_rust_array("B2", b2, f"[{HIDDEN_DIM}]")}

/// Layer 3 (output) weights: [{OUTPUT_DIM} output neurons, {HIDDEN_DIM} hidden neurons]
{format_rust_array("W3", w3, f"[{OUTPUT_DIM}, {HIDDEN_DIM}]", neuron_labels_w3)}

{format_rust_array("B3", b3, f"[{OUTPUT_DIM}]")}
"""

    with open(filepath, "w") as f:
        f.write(content)
    print(f"\nRust weights written to {filepath}")


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

def compute_decision(pred_class):
    """Map class to decision: 0,1 -> allow; 2,3 -> deny."""
    return "deny" if pred_class >= 2 else "allow"


def print_report(model, X, y, label=""):
    """Print per-class precision/recall/F1 and confusion matrix."""
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X))
        pred = out.argmax(dim=1).numpy()

    print(f"\n{'='*60}")
    print(f"  {label} Report")
    print(f"{'='*60}")

    # Exact match accuracy
    exact_acc = (pred == y).mean()
    print(f"\nExact match accuracy: {exact_acc:.4f} ({(pred == y).sum()}/{len(y)})")

    # Decision accuracy (allow vs deny)
    pred_decisions = np.array([compute_decision(p) for p in pred])
    true_decisions = np.array([compute_decision(t) for t in y])
    decision_acc = (pred_decisions == true_decisions).mean()
    print(f"Decision accuracy:    {decision_acc:.4f} ({(pred_decisions == true_decisions).sum()}/{len(y)})")

    # Per-class precision/recall/F1
    print(f"\n{'Class':<12} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>8}")
    print("-" * 42)
    for c in range(OUTPUT_DIM):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        support = (y == c).sum()
        print(f"{CLASS_NAMES[c]:<12} {prec:>6.3f} {recall:>6.3f} {f1:>6.3f} {support:>8d}")

    # Confusion matrix
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>12}", end="")
    for c in range(OUTPUT_DIM):
        print(f" {CLASS_NAMES[c]:>10}", end="")
    print()
    for true_c in range(OUTPUT_DIM):
        print(f"{CLASS_NAMES[true_c]:>12}", end="")
        for pred_c in range(OUTPUT_DIM):
            count = ((y == true_c) & (pred == pred_c)).sum()
            print(f" {count:>10}", end="")
        print()

    return exact_acc, decision_acc


def print_feature_importance(quantized):
    """Print feature importance based on W1 weight magnitudes."""
    w1 = quantized["fc1.weight"]  # [32, 28]
    importance = np.abs(w1).mean(axis=0)  # mean abs weight per feature

    print(f"\n{'='*60}")
    print("  Feature Importance (mean |W1| per feature)")
    print(f"{'='*60}")

    ranked = np.argsort(-importance)
    for rank, idx in enumerate(ranked):
        bar = "█" * int(importance[idx] / importance.max() * 30)
        print(f"  {rank+1:2d}. [{idx:2d}] {FEATURE_NAMES[idx]:<30} {importance[idx]:6.1f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("Generating synthetic dataset...")
    X, y = generate_dataset(rng, n_total=2000)
    print(f"  Total samples: {len(y)}")
    for c in range(OUTPUT_DIM):
        print(f"  {CLASS_NAMES[c]}: {(y == c).sum()}")

    # 80/20 split
    n = len(y)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"\n  Train: {len(y_train)}, Val: {len(y_val)}")
    print(f"\nTraining {INPUT_DIM}->{HIDDEN_DIM}->{HIDDEN_DIM}->{OUTPUT_DIM} MLP...")

    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate on full dataset and validation set
    print_report(model, X_train, y_train, "Training")
    val_exact, val_decision = print_report(model, X_val, y_val, "Validation")

    # Quantize
    print("\nQuantizing weights (scale=128)...")
    quantized = quantize_weights(model)
    for name, arr in quantized.items():
        print(f"  {name}: shape={arr.shape}, range=[{arr.min()}, {arr.max()}]")

    # Feature importance
    print_feature_importance(quantized)

    # Export
    export_rust_weights(quantized, "training/weights.rs")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Validation exact accuracy:    {val_exact:.4f}")
    print(f"  Validation decision accuracy: {val_decision:.4f}")
    total_params = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    print(f"  Total parameters:             {total_params}")
    print(f"  Target: decision accuracy >= 0.94")

    if val_decision >= 0.94:
        print("  ✓ TARGET MET")
    else:
        print("  ✗ TARGET NOT MET — consider adjusting hyperparameters")


if __name__ == "__main__":
    main()
