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
from sklearn.model_selection import StratifiedKFold

from dataset import (
    CLASS_NAMES,
    CLASS_NAMES_3,
    NUM_FEATURES,
    SkillDataset,
    generate_dataset,
    generate_dangerous_augmentation,
    load_real_dataset,
    save_dataset_jsonl,
)
from model import SkillSafetyMLP


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
    adv_epsilon: float = 2.0,
    adv_ratio: float = 0.3,
    verbose: bool = True,
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

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
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

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch+1:3d}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
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
) -> tuple[dict, dict]:
    """Train with k-fold stratified cross-validation.

    Returns (best_fold_result, summary).
    """
    if class_names is None:
        class_names = CLASS_NAMES if num_classes == 4 else CLASS_NAMES_3

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    best_fold_idx = 0
    best_fold_acc = 0.0

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
            verbose=verbose,
        )

        fold_results.append(result)

        if verbose:
            print(f"  Best val accuracy: {result['best_val_acc']:.4f}")
            report = result["classification_report"]
            for cls_name in class_names:
                if cls_name in report:
                    p = report[cls_name]["precision"]
                    r = report[cls_name]["recall"]
                    f1 = report[cls_name]["f1-score"]
                    print(f"  {cls_name:10s}: P={p:.3f} R={r:.3f} F1={f1:.3f}")

        if result["best_val_acc"] > best_fold_acc:
            best_fold_acc = result["best_val_acc"]
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

    # Summary
    accs = [r["best_val_acc"] for r in fold_results]
    summary = {
        "n_folds": n_folds,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "best_fold": best_fold_idx,
        "best_accuracy": best_fold_acc,
        "per_fold_accuracy": accs,
        "per_class_metrics": per_class_metrics,
    }

    if verbose:
        print(f"\n=== Cross-Validation Summary ===")
        print(f"Mean accuracy: {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")
        print(f"Best fold: {summary['best_fold'] + 1} ({summary['best_accuracy']:.4f})")
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
        verbose=verbose,
    )

    # Save model
    model_path = Path("data/model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), model_path)
    if verbose:
        print(f"Saved model to {model_path}")
        print(f"Full-data accuracy: {final_result['best_val_acc']:.4f}")

    # Export if requested
    if args.export:
        from export_weights import export_to_rust

        export_to_rust(final_model, num_classes=num_classes, verbose=verbose)

    return 0 if summary["mean_accuracy"] >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
