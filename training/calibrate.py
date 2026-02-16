"""
Temperature calibration for softmax scoring.

Grid-searches over temperature values to minimize Expected Calibration Error (ECE)
on the validation set.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import NUM_FEATURES, generate_dataset, CLASS_NAMES
from model import SkillSafetyMLP


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature-scaled softmax."""
    scaled = logits / temperature
    # Numerical stability
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well the predicted probabilities match empirical accuracy.
    Lower is better.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = confidences[mask].mean()
        avg_accuracy = (predictions[mask] == labels[mask]).mean()
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return ece


def calibrate_temperature(
    model: SkillSafetyMLP,
    features: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True,
) -> tuple[float, float]:
    """Find optimal softmax temperature via grid search.

    Returns (optimal_temperature, best_ece).
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = model(x).numpy()

    # Grid search
    temperatures = list(np.arange(0.1, 1.05, 0.05)) + [1.5, 2.0, 3.0, 5.0]
    best_temp = 1.0
    best_ece = float("inf")
    results = []

    for temp in temperatures:
        probs = softmax_with_temperature(logits, temp)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracy = (predictions == labels).mean()
        ece = expected_calibration_error(confidences, predictions, labels)

        results.append((temp, ece, accuracy))

        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    if verbose:
        print("=== Temperature Calibration ===")
        print(f"{'Temp':>6s}  {'ECE':>8s}  {'Accuracy':>8s}")
        for temp, ece, acc in sorted(results, key=lambda x: x[1])[:10]:
            marker = " *" if abs(temp - best_temp) < 0.01 else ""
            print(f"{temp:6.2f}  {ece:8.4f}  {acc:8.4f}{marker}")
        print(f"\nOptimal temperature: {best_temp:.2f} (ECE={best_ece:.4f})")

    return best_temp, best_ece


def compute_entropy_threshold(
    model: SkillSafetyMLP,
    features: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    target_flag_rate: float = 0.05,
    verbose: bool = True,
) -> float:
    """Compute entropy threshold for abstaining (flagging uncertain predictions).

    Finds a threshold such that ~target_flag_rate of predictions would be flagged.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = model(x).numpy()

    probs = softmax_with_temperature(logits, temperature)

    # Compute entropy for each sample
    entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    max_entropy = np.log(4)  # max entropy for 4 classes

    # Normalize to [0, 1]
    norm_entropies = entropies / max_entropy

    # Find threshold at target_flag_rate percentile from the top
    threshold = np.percentile(norm_entropies, (1.0 - target_flag_rate) * 100)

    if verbose:
        n_flagged = (norm_entropies > threshold).sum()
        print(f"\n=== Entropy Threshold ===")
        print(f"Target flag rate: {target_flag_rate:.1%}")
        print(f"Threshold: {threshold:.4f} (normalized entropy)")
        print(f"Would flag: {n_flagged}/{len(labels)} ({n_flagged/len(labels):.1%})")

        # Check accuracy of flagged vs unflagged
        predictions = probs.argmax(axis=1)
        flagged = norm_entropies > threshold
        unflagged = ~flagged
        if unflagged.sum() > 0:
            print(
                f"Accuracy on unflagged: {(predictions[unflagged] == labels[unflagged]).mean():.4f}"
            )
        if flagged.sum() > 0:
            print(
                f"Accuracy on flagged:   {(predictions[flagged] == labels[flagged]).mean():.4f}"
            )

    return threshold


def main():
    parser = argparse.ArgumentParser(description="Calibrate softmax temperature")
    parser.add_argument(
        "--model", type=str, default="data/model.pt", help="Path to model checkpoint"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    model = SkillSafetyMLP(input_dim=NUM_FEATURES, qat=False)
    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    features, labels = generate_dataset(seed=args.seed)

    best_temp, best_ece = calibrate_temperature(model, features, labels, verbose=verbose)
    entropy_threshold = compute_entropy_threshold(
        model, features, labels, best_temp, verbose=verbose
    )

    # Save results
    results = {
        "optimal_temperature": best_temp,
        "ece": best_ece,
        "entropy_threshold": entropy_threshold,
    }
    output_path = Path("data/calibration.json")
    output_path.write_text(__import__("json").dumps(results, indent=2))
    if verbose:
        print(f"\nSaved calibration results to {output_path}")


if __name__ == "__main__":
    main()
