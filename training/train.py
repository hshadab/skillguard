"""
Training loop for SkillGuard skill safety classifier.

Features:
- 5-fold stratified cross-validation
- Adam optimizer with cross-entropy loss
- Early stopping
- Per-class precision/recall/F1
- Quantization-Aware Training (QAT)
- FGSM adversarial training
- Exports best model weights
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from dataset import (
    CLASS_NAMES,
    CLASS_NAMES_3,
    NUM_FEATURES,
    SkillDataset,
    generate_dataset,
    generate_dangerous_augmentation,
    load_real_dataset,
    oversample_dangerous_smote,
    save_dataset_jsonl,
)
from model import SkillSafetyMLP


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    Down-weights easy/majority-class examples so the model focuses on hard
    boundary cases (e.g., DANGEROUS vs CAUTION). Combines with per-class
    alpha weights for class-imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce_loss).mean()


class DangerSensitiveLoss(nn.Module):
    """Cross-entropy with asymmetric DANGEROUS false-negative penalty.

    Applies an extra multiplicative penalty when a true DANGEROUS sample
    is predicted as non-DANGEROUS, forcing the model to prioritize
    DANGEROUS recall over overall accuracy.
    """

    def __init__(self, class_weights=None, danger_fn_weight=10.0):
        super().__init__()
        self.danger_fn_weight = danger_fn_weight
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.class_weights, reduction="none"
        )
        # Extra penalty when true DANGEROUS is predicted as non-DANGEROUS
        pred = logits.argmax(dim=1)
        dangerous_mask = targets == 2
        false_negative_mask = dangerous_mask & (pred != 2)
        penalty = torch.ones_like(ce)
        penalty[false_negative_mask] = self.danger_fn_weight
        return (penalty * ce).mean()


def compute_safety_metric(val_pred, val_y, num_classes=3, danger_fn_weight=0.0):
    """Compute recall-weighted metric for early stopping.

    When danger_fn_weight > 0, shift weights to prioritize DANGEROUS recall
    (SAFE=0.15, CAUTION=0.30, DANGEROUS=0.55). Otherwise use balanced weights
    (SAFE=0.25, CAUTION=0.40, DANGEROUS=0.35).
    """
    recalls = []
    for c in range(num_classes):
        mask = val_y == c
        if mask.sum() > 0:
            recalls.append(float((val_pred[mask] == c).sum()) / float(mask.sum()))
        else:
            recalls.append(0.0)
    if danger_fn_weight > 0:
        # DANGEROUS-priority mode: checkpoint selection favors DANGEROUS recall
        weights = [0.15, 0.30, 0.55]
    else:
        # Balanced mode: SAFE=0.25, CAUTION=0.40, DANGEROUS=0.35
        weights = [0.25, 0.40, 0.35]
    if num_classes > len(weights):
        weights = [1.0 / num_classes] * num_classes
    return sum(w * r for w, r in zip(weights[:num_classes], recalls))


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 2.0,
    criterion: nn.Module = None,
) -> torch.Tensor:
    """Generate FGSM adversarial examples.

    Args:
        model: The model to attack
        x: Input features (batch)
        y: True labels
        epsilon: Perturbation magnitude (in feature space, [0, 128])
        criterion: Loss function

    Returns:
        Perturbed input tensor
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = criterion(output, y)
    loss.backward()

    # FGSM: perturb in the direction of the gradient sign
    perturbation = epsilon * x_adv.grad.sign()
    x_perturbed = x_adv + perturbation

    # Clip to valid range [0, 128]
    x_perturbed = torch.clamp(x_perturbed, 0.0, 128.0)

    return x_perturbed.detach()


def train_one_fold(
    model: SkillSafetyMLP,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    class_names: list[str] | None = None,
    class_weights: torch.Tensor | None = None,
    epochs: int = 200,
    lr: float = 0.001,
    patience: int = 30,
    adversarial: bool = True,
    adv_epsilon: float = 0.5,
    adv_ratio: float = 0.3,
    focal_gamma: float = 2.0,
    scheduler_type: str = "plateau",
    verbose: bool = True,
    danger_fn_weight: float = 0.0,
    num_classes: int = 3,
) -> dict:
    """Train model on one fold with early stopping.

    Returns dict with best metrics and model state_dict.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    train_dataset = SkillDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    if danger_fn_weight > 0:
        criterion = DangerSensitiveLoss(
            class_weights=class_weights, danger_fn_weight=danger_fn_weight
        )
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-5
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

    best_val_acc = 0.0
    best_val_metric = 0.0
    best_state = None
    epochs_without_improvement = 0

    # Track current training data for hard negative mining
    train_features_current = train_features
    train_labels_current = train_labels

    for epoch in range(epochs):
        # Hard negative mining: every 50 epochs, find misclassified DANGEROUS
        # samples, duplicate them with noise, and rebuild the dataloader.
        if epoch > 0 and epoch % 50 == 0 and adversarial:
            model.eval()
            with torch.no_grad():
                all_x = torch.tensor(train_features_current, dtype=torch.float32)
                all_y = torch.tensor(train_labels_current, dtype=torch.long)
                pred = model(all_x).argmax(1)

                aug_xs = [all_x]
                aug_ys = [all_y]

                # Find DANGEROUS false negatives (DANGEROUS predicted as non-DANGEROUS)
                danger_mask = (all_y == 2) & (pred != 2)
                if danger_mask.sum() > 0:
                    hard_neg_x = all_x[danger_mask]
                    hard_neg_y = all_y[danger_mask]
                    # More copies when in danger-priority mode
                    n_copies = 4 if danger_fn_weight > 0 else 2
                    for _ in range(n_copies - 1):
                        noise = torch.randn_like(hard_neg_x) * 2.0
                        hard_neg_noisy = torch.clamp(hard_neg_x + noise, 0, 128)
                        aug_xs.append(hard_neg_noisy)
                        aug_ys.append(hard_neg_y)
                    aug_xs.append(hard_neg_x)
                    aug_ys.append(hard_neg_y)
                    if verbose:
                        print(f"  Epoch {epoch}: mined {int(danger_mask.sum())} hard DANGEROUS negatives (x{n_copies})")

                # Find CAUTION false negatives (CAUTION predicted as SAFE)
                # Skip in danger-priority mode to avoid diluting DANGEROUS signal
                if danger_fn_weight == 0:
                    caution_fn_mask = (all_y == 1) & (pred == 0)
                    if caution_fn_mask.sum() > 0:
                        hard_caut_x = all_x[caution_fn_mask]
                        hard_caut_y = all_y[caution_fn_mask]
                        noise_c = torch.randn_like(hard_caut_x) * 2.0
                        hard_caut_noisy = torch.clamp(hard_caut_x + noise_c, 0, 128)
                        aug_xs.extend([hard_caut_x, hard_caut_noisy])
                        aug_ys.extend([hard_caut_y, hard_caut_y])
                        if verbose:
                            print(f"  Epoch {epoch}: mined {int(caution_fn_mask.sum())} hard CAUTION negatives")

                if len(aug_xs) > 1:
                    aug_x = torch.cat(aug_xs)
                    aug_y = torch.cat(aug_ys)
                    train_features_current = aug_x.numpy()
                    train_labels_current = aug_y.numpy()
                    train_dataset = SkillDataset(train_features_current, train_labels_current)
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=64, shuffle=True
                    )

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Standard forward pass
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # Adversarial training: add FGSM-perturbed examples
            if adversarial and adv_ratio > 0:
                n_adv = max(1, int(len(batch_x) * adv_ratio))
                adv_x = fgsm_attack(
                    model, batch_x[:n_adv], batch_y[:n_adv], adv_epsilon, criterion
                )
                adv_output = model(adv_x)
                adv_loss = criterion(adv_output, batch_y[:n_adv])
                loss = loss + adv_ratio * adv_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(val_features, dtype=torch.float32)
            val_y = torch.tensor(val_labels, dtype=torch.long)
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y).item()
            _, val_pred = val_output.max(1)
            val_acc = val_pred.eq(val_y).sum().item() / len(val_y)
            val_metric = compute_safety_metric(val_pred, val_y, num_classes=num_classes, danger_fn_weight=danger_fn_weight)

        if scheduler_type == "cosine":
            scheduler.step(epoch + val_loss * 0)  # CosineAnnealing uses epoch count
        else:
            scheduler.step(val_loss)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch+1:3d}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"val_metric={val_metric:.4f}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Final evaluation with best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(val_features, dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.long)
        val_output = model(val_x)
        _, val_pred = val_output.max(1)
        val_pred_np = val_pred.numpy()
        val_y_np = val_y.numpy()

    report = classification_report(
        val_y_np, val_pred_np, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(val_y_np, val_pred_np)

    return {
        "best_val_acc": best_val_acc,
        "best_val_metric": best_val_metric,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "state_dict": best_state,
    }


def train_kfold(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 4,
    class_names: list[str] | None = None,
    class_weights: torch.Tensor | None = None,
    n_folds: int = 5,
    seed: int = 42,
    epochs: int = 200,
    adversarial: bool = True,
    verbose: bool = True,
    patience: int = 30,
    adv_epsilon: float = 0.5,
    focal_gamma: float = 2.0,
    scheduler_type: str = "plateau",
    danger_fn_weight: float = 0.0,
) -> tuple[dict, dict]:
    """Train with k-fold stratified cross-validation.

    Returns (best_fold_result, summary).
    """
    if class_names is None:
        class_names = CLASS_NAMES if num_classes == 4 else CLASS_NAMES_3

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    best_fold_idx = 0
    best_fold_metric = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]

        model = SkillSafetyMLP(input_dim=NUM_FEATURES, num_classes=num_classes, qat=True)
        torch.manual_seed(seed + fold_idx)

        result = train_one_fold(
            model,
            train_features,
            train_labels,
            val_features,
            val_labels,
            class_names=class_names,
            class_weights=class_weights,
            epochs=epochs,
            patience=patience,
            adversarial=adversarial,
            adv_epsilon=adv_epsilon,
            focal_gamma=focal_gamma,
            scheduler_type=scheduler_type,
            verbose=verbose,
            danger_fn_weight=danger_fn_weight,
            num_classes=num_classes,
        )

        fold_results.append(result)

        if verbose:
            print(f"  Best val acc: {result['best_val_acc']:.4f}  metric: {result['best_val_metric']:.4f}")
            report = result["classification_report"]
            for cls_name in class_names:
                if cls_name in report:
                    p = report[cls_name]["precision"]
                    r = report[cls_name]["recall"]
                    f1 = report[cls_name]["f1-score"]
                    print(f"  {cls_name:10s}: P={p:.3f} R={r:.3f} F1={f1:.3f}")

        if result["best_val_metric"] > best_fold_metric:
            best_fold_metric = result["best_val_metric"]
            best_fold_idx = fold_idx

    # Aggregate per-class metrics across folds
    per_class_metrics = {}
    for cls_name in class_names:
        precisions = []
        recalls = []
        f1s = []
        for r in fold_results:
            report = r["classification_report"]
            if cls_name in report:
                precisions.append(report[cls_name]["precision"])
                recalls.append(report[cls_name]["recall"])
                f1s.append(report[cls_name]["f1-score"])
        per_class_metrics[cls_name] = {
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }

    # Aggregate confusion matrices across folds (sum, not average)
    n_cls = len(class_names)
    agg_cm = np.zeros((n_cls, n_cls), dtype=int)
    for r in fold_results:
        agg_cm += np.array(r["confusion_matrix"])

    # Binary DANGEROUS-vs-rest metrics from aggregated confusion matrix
    dangerous_idx = class_names.index("DANGEROUS") if "DANGEROUS" in class_names else n_cls - 1
    tp_d = int(agg_cm[dangerous_idx, dangerous_idx])
    fn_d = int(agg_cm[dangerous_idx, :].sum() - tp_d)
    fp_d = int(agg_cm[:, dangerous_idx].sum() - tp_d)
    tn_d = int(agg_cm.sum() - tp_d - fn_d - fp_d)
    binary_precision = tp_d / max(tp_d + fp_d, 1)
    binary_recall = tp_d / max(tp_d + fn_d, 1)
    binary_f1 = 2 * binary_precision * binary_recall / max(binary_precision + binary_recall, 1e-9)
    binary_accuracy = (tp_d + tn_d) / max(agg_cm.sum(), 1)
    false_alarm_rate = fp_d / max(fp_d + tn_d, 1)  # safe/caution incorrectly denied

    # Summary
    accs = [r["best_val_acc"] for r in fold_results]
    metrics = [r["best_val_metric"] for r in fold_results]
    summary = {
        "n_folds": n_folds,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_safety_metric": float(np.mean(metrics)),
        "best_fold": best_fold_idx,
        "best_accuracy": float(fold_results[best_fold_idx]["best_val_acc"]),
        "best_safety_metric": best_fold_metric,
        "per_fold_accuracy": accs,
        "per_fold_metric": metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": agg_cm.tolist(),
        "confusion_matrix_labels": class_names,
        "binary_dangerous_vs_rest": {
            "accuracy": float(binary_accuracy),
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1": float(binary_f1),
            "false_alarm_rate": float(false_alarm_rate),
            "miss_rate": float(1.0 - binary_recall),
            "tp": tp_d,
            "fn": fn_d,
            "fp": fp_d,
            "tn": tn_d,
        },
    }

    if verbose:
        print(f"\n=== Cross-Validation Summary ===")
        print(f"Mean accuracy: {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")
        print(f"Mean safety metric: {summary['mean_safety_metric']:.4f}")
        print(f"Best fold: {summary['best_fold'] + 1} (metric={best_fold_metric:.4f}, acc={summary['best_accuracy']:.4f})")
        print(f"\nPer-class metrics (mean +/- std across {n_folds} folds):")
        print(f"  {'Class':10s}  {'Precision':>12s}  {'Recall':>12s}  {'F1':>12s}")
        for cls_name in class_names:
            m = per_class_metrics[cls_name]
            print(
                f"  {cls_name:10s}  "
                f"{m['precision_mean']:.3f}+/-{m['precision_std']:.3f}  "
                f"{m['recall_mean']:.3f}+/-{m['recall_std']:.3f}  "
                f"{m['f1_mean']:.3f}+/-{m['f1_std']:.3f}"
            )
        print(f"\nConfusion matrix (rows=actual, cols=predicted):")
        header = "            " + "  ".join(f"{c:>10s}" for c in class_names)
        print(header)
        for i, cls_name in enumerate(class_names):
            row = "  ".join(f"{agg_cm[i, j]:>10d}" for j in range(n_cls))
            print(f"  {cls_name:10s}{row}")
        bd = summary["binary_dangerous_vs_rest"]
        print(f"\nBinary DANGEROUS-vs-rest:")
        print(f"  Catch rate (recall):  {bd['recall']:.1%}  ({bd['tp']} caught, {bd['fn']} missed)")
        print(f"  False alarm rate:     {bd['false_alarm_rate']:.1%}  ({bd['fp']} safe/caution denied)")
        print(f"  Precision:            {bd['precision']:.1%}")
        print(f"  Binary F1:            {bd['f1']:.3f}")
        print(f"  Binary accuracy:      {bd['accuracy']:.1%}")

    return fold_results[best_fold_idx], summary


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    # Avoid division by zero
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train SkillGuard safety classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-per-class", type=int, default=125)
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument("--export", action="store_true", help="Export weights to Rust")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "real"],
        help="Dataset type: synthetic (4-class) or real (3-class LLM-labeled)"
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Number of output classes (default: 4 for synthetic, 3 for real)"
    )
    parser.add_argument(
        "--real-labels", type=str, default="training/real-labels.json",
        help="Path to real labeled dataset (used with --dataset real)"
    )
    parser.add_argument(
        "--augment-dangerous", type=int, default=0, metavar="N",
        help="Generate N synthetic DANGEROUS samples and mix with real data. "
             "Useful when real DANGEROUS samples are scarce (e.g., <40)."
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early-stopping patience (default: 30 for small datasets, 50 for large)"
    )
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.0, metavar="F",
        help="Fraction of data to hold out as a final test set (0.0 = disabled). "
             "When >0, a stratified split is done before CV: (1-F) for train/CV, "
             "F for holdout. The final model is evaluated on holdout separately."
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss gamma parameter (default 2.0). Lower values (1.0-1.5) "
             "reduce down-weighting of hard examples, useful when many CAUTION "
             "samples are genuinely difficult."
    )
    parser.add_argument(
        "--adv-epsilon", type=float, default=0.5,
        help="FGSM adversarial perturbation magnitude (default 0.5). "
             "Higher values create stronger adversarial examples."
    )
    parser.add_argument(
        "--scheduler", type=str, default="plateau",
        choices=["plateau", "cosine"],
        help="Learning rate scheduler: 'plateau' (ReduceLROnPlateau) or "
             "'cosine' (CosineAnnealingWarmRestarts). Cosine finds flatter "
             "minima on small datasets."
    )
    parser.add_argument(
        "--danger-fn-weight", type=float, default=0.0,
        help="Asymmetric DANGEROUS false-negative penalty weight (default 0 = use FocalLoss). "
             "When >0, uses DangerSensitiveLoss instead of FocalLoss with this penalty "
             "for DANGEROUS samples predicted as non-DANGEROUS."
    )
    parser.add_argument(
        "--oversample-dangerous", type=float, default=0.0, metavar="RATIO",
        help="SMOTE-like oversampling target ratio for DANGEROUS class (default 0 = disabled). "
             "E.g., 0.25 means generate synthetic samples until DANGEROUS is ~25%% of dataset."
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # Determine num_classes
    if args.num_classes is not None:
        num_classes = args.num_classes
    elif args.dataset == "real":
        num_classes = 3
    else:
        num_classes = 4

    class_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES

    if args.dataset == "real":
        if verbose:
            print(f"Loading real dataset from {args.real_labels}...")
        features, labels = load_real_dataset(args.real_labels)

        if len(features) == 0:
            print("ERROR: No samples loaded from real dataset", file=sys.stderr)
            return 1

        # Augment with synthetic samples if requested
        if args.augment_dangerous > 0:
            aug_features, aug_labels = generate_dangerous_augmentation(
                n=args.augment_dangerous, seed=args.seed
            )
            features = np.concatenate([features, aug_features])
            labels = np.concatenate([labels, aug_labels])
            n_aug_dangerous = (aug_labels == 2).sum()
            n_aug_safe = (aug_labels == 0).sum()
            if verbose:
                print(f"Augmented with {n_aug_dangerous} DANGEROUS + {n_aug_safe} SAFE synthetic samples")

        # SMOTE-like oversampling for DANGEROUS class
        if args.oversample_dangerous > 0:
            n_before = (labels == 2).sum()
            features, labels = oversample_dangerous_smote(
                features, labels, target_ratio=args.oversample_dangerous, seed=args.seed
            )
            n_after = (labels == 2).sum()
            if verbose:
                print(f"SMOTE oversampling: DANGEROUS {n_before} -> {n_after} "
                      f"(target ratio {args.oversample_dangerous})")

        if verbose:
            print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
            for i, name in enumerate(class_names):
                print(f"  {name}: {(labels == i).sum()}")

        # Compute class weights for imbalanced real data
        class_weights = compute_class_weights(labels, num_classes)
        if verbose:
            print(f"Class weights: {class_weights.tolist()}")
    else:
        if verbose:
            print(f"Generating synthetic dataset (seed={args.seed})...")

        features, labels = generate_dataset(
            n_per_class=args.n_per_class,
            seed=args.seed,
        )
        class_weights = None

        # Save dataset
        save_dataset_jsonl(features, labels)

        if verbose:
            print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
            for i, name in enumerate(class_names):
                count = (labels == i).sum()
                if count > 0:
                    print(f"  {name}: {count}")

    # Holdout split: set aside a stratified fraction for final evaluation
    holdout_features = None
    holdout_labels = None
    if args.holdout_fraction > 0:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.holdout_fraction, random_state=args.seed
        )
        train_idx, holdout_idx = next(sss.split(features, labels))
        holdout_features = features[holdout_idx]
        holdout_labels = labels[holdout_idx]
        features = features[train_idx]
        labels = labels[train_idx]
        if verbose:
            print(f"Holdout split: {len(holdout_labels)} holdout, {len(labels)} train/CV")
            for i, name in enumerate(class_names):
                n_h = (holdout_labels == i).sum()
                n_t = (labels == i).sum()
                print(f"  {name}: {n_t} train/CV, {n_h} holdout")

        # Recompute class weights on the train/CV portion only
        if args.dataset == "real":
            class_weights = compute_class_weights(labels, num_classes)

    # Determine early-stopping patience: default 50 for large datasets, 30 otherwise
    patience = args.patience
    if patience is None:
        patience = 50 if len(features) >= 300 else 30
    if verbose:
        print(f"Early-stopping patience: {patience}")

    # Train
    torch.manual_seed(args.seed)
    best_result, summary = train_kfold(
        features,
        labels,
        num_classes=num_classes,
        class_names=class_names,
        class_weights=class_weights,
        n_folds=args.folds,
        seed=args.seed,
        epochs=args.epochs,
        adversarial=not args.no_adversarial,
        verbose=verbose,
        patience=patience,
        adv_epsilon=args.adv_epsilon,
        focal_gamma=args.focal_gamma,
        scheduler_type=args.scheduler,
        danger_fn_weight=args.danger_fn_weight,
    )

    # Save summary
    summary_path = Path("data/training_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"\nSaved training summary to {summary_path}")

    # Retrain on full dataset with best hyperparameters
    if verbose:
        print("\nRetraining on full dataset...")
    final_model = SkillSafetyMLP(input_dim=NUM_FEATURES, num_classes=num_classes, qat=True)
    torch.manual_seed(args.seed)

    final_result = train_one_fold(
        final_model,
        features,
        labels,
        features,  # Validate on training data (we already have CV results)
        labels,
        class_names=class_names,
        class_weights=class_weights,
        epochs=args.epochs,
        adversarial=not args.no_adversarial,
        adv_epsilon=args.adv_epsilon,
        focal_gamma=args.focal_gamma,
        scheduler_type=args.scheduler,
        verbose=verbose,
        danger_fn_weight=args.danger_fn_weight,
        num_classes=num_classes,
    )

    # Save model
    model_path = Path("data/model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), model_path)
    if verbose:
        print(f"Saved model to {model_path}")
        print(f"Full-data accuracy: {final_result['best_val_acc']:.4f}")

    # Holdout evaluation
    if holdout_features is not None and len(holdout_features) > 0:
        if verbose:
            print(f"\n=== Holdout Evaluation ({len(holdout_labels)} samples) ===")
        final_model.eval()
        with torch.no_grad():
            h_x = torch.tensor(holdout_features, dtype=torch.float32)
            h_y = torch.tensor(holdout_labels, dtype=torch.long)
            h_output = final_model(h_x)
            _, h_pred = h_output.max(1)
            h_pred_np = h_pred.numpy()
            h_y_np = h_y.numpy()

        holdout_acc = (h_pred_np == h_y_np).sum() / len(h_y_np)
        holdout_report = classification_report(
            h_y_np, h_pred_np, target_names=class_names, output_dict=True
        )
        holdout_cm = confusion_matrix(h_y_np, h_pred_np)

        summary["holdout_accuracy"] = float(holdout_acc)
        summary["holdout_report"] = holdout_report
        summary["holdout_confusion_matrix"] = holdout_cm.tolist()

        if verbose:
            print(f"Holdout accuracy: {holdout_acc:.4f}")
            print(f"  {'Class':10s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}")
            for cls_name in class_names:
                if cls_name in holdout_report:
                    p = holdout_report[cls_name]["precision"]
                    r = holdout_report[cls_name]["recall"]
                    f1 = holdout_report[cls_name]["f1-score"]
                    print(f"  {cls_name:10s}  {p:9.3f}  {r:9.3f}  {f1:9.3f}")

        # Re-save summary with holdout metrics
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Export if requested
    if args.export:
        from export_weights import export_to_rust

        export_to_rust(final_model, num_classes=num_classes, verbose=verbose)

    # With DANGEROUS-recall optimization, accuracy may drop to ~55%
    # but safety metric should be high. Accept lower accuracy threshold.
    min_acc = 0.40 if args.danger_fn_weight > 0 else 0.90
    return 0 if summary["mean_accuracy"] >= min_acc else 1


if __name__ == "__main__":
    sys.exit(main())
