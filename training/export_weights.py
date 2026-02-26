"""
Export PyTorch model weights to Rust fixed-point i32 const arrays.

Converts float32 weights to i32 with scale=7 (multiply by 128),
validates roundtrip accuracy, and outputs Rust source code.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import CLASS_NAMES, CLASS_NAMES_3, NUM_FEATURES, generate_dataset, load_real_dataset, SkillDataset
from model import SkillSafetyMLP


SCALE = 128  # 2^7


def float_to_fixed(weight: np.ndarray) -> np.ndarray:
    """Convert float32 weights to i32 fixed-point (scale=128)."""
    return np.round(weight * SCALE).astype(np.int32)


def fixed_to_float(fixed: np.ndarray) -> np.ndarray:
    """Convert i32 fixed-point back to float32."""
    return fixed.astype(np.float32) / SCALE


def rust_i32_forward(features_i32: np.ndarray, state: dict) -> np.ndarray:
    """Simulate the exact Rust i32 inference path for a batch of samples.

    Matches model.rs: matmul → (result + 64) / 128 → + bias → relu → repeat.
    All arithmetic uses i32 integer operations.

    Args:
        features_i32: (N, 45) array of i32 feature values
        state: model state_dict with float weights

    Returns:
        (N, num_classes) array of i32 logits
    """
    # Quantize all weights and biases to i32
    w1 = float_to_fixed(state["fc1.weight"].numpy())  # [56, 45]
    b1 = float_to_fixed(state["fc1.bias"].numpy())     # [56]
    w2 = float_to_fixed(state["fc2.weight"].numpy())  # [40, 56]
    b2 = float_to_fixed(state["fc2.bias"].numpy())     # [40]
    w3 = float_to_fixed(state["fc3.weight"].numpy())  # [C, 40]
    b3 = float_to_fixed(state["fc3.bias"].numpy())     # [C]

    x = features_i32.astype(np.int64)  # use i64 to avoid overflow

    # Layer 1: [N, 45] @ [45, 56] -> [N, 56]
    mm1 = x @ w1.astype(np.int64).T           # [N, 56]
    mm1 = (mm1 + 64) // 128                    # integer floor division
    mm1 = mm1 + b1.astype(np.int64)            # add bias
    mm1 = np.maximum(mm1, 0)                    # ReLU

    # Layer 2: [N, 56] @ [56, 40] -> [N, 40]
    mm2 = mm1 @ w2.astype(np.int64).T         # [N, 40]
    mm2 = (mm2 + 64) // 128
    mm2 = mm2 + b2.astype(np.int64)
    mm2 = np.maximum(mm2, 0)                    # ReLU

    # Layer 3: [N, 40] @ [40, C] -> [N, C]
    mm3 = mm2 @ w3.astype(np.int64).T         # [N, C]
    mm3 = (mm3 + 64) // 128
    mm3 = mm3 + b3.astype(np.int64)
    # No ReLU on output

    return mm3.astype(np.int32)


def validate_roundtrip(
    model: SkillSafetyMLP,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 4,
    verbose: bool = True,
) -> tuple[float, float]:
    """Validate that fixed-point conversion preserves classification accuracy.

    Simulates the EXACT Rust i32 inference path:
      matmul(x_i32, w_i32) → (result + 64) // 128 → + bias_i32 → relu → repeat

    Returns (float_accuracy, fixed_accuracy).
    """
    model.eval()

    # Float inference (QAT model)
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        float_output = model(x)
        _, float_pred = float_output.max(1)
        float_acc = float_pred.eq(y).sum().item() / len(y)

    # True i32 integer inference (matches Rust exactly)
    features_i32 = np.round(features).astype(np.int32)
    state = model.state_dict()
    i32_logits = rust_i32_forward(features_i32, state)
    fixed_pred_np = np.argmax(i32_logits, axis=1)
    y_np = labels.astype(np.int64)
    fixed_acc = (fixed_pred_np == y_np).sum() / len(y_np)

    # Decision match
    deny_threshold = 2
    float_decisions = (float_pred.numpy() >= deny_threshold).astype(int)
    fixed_decisions = (fixed_pred_np >= deny_threshold).astype(int)
    decision_match = (float_decisions == fixed_decisions).sum() / len(y_np)

    if verbose:
        print(f"\n=== Roundtrip Validation (true i32 simulation) ===")
        print(f"Float (QAT) accuracy:  {float_acc:.4f}")
        print(f"Fixed-point i32 accuracy: {fixed_acc:.4f}")
        print(f"Decision match:        {decision_match:.4f}")
        mismatches = (float_pred.numpy() != fixed_pred_np).sum()
        print(f"Classification diff:   {mismatches} samples")

        # Show per-class i32 accuracy
        class_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES
        for cls_idx, cls_name in enumerate(class_names):
            mask = (y_np == cls_idx)
            if mask.sum() > 0:
                cls_acc = (fixed_pred_np[mask] == cls_idx).sum() / mask.sum()
                float_cls_acc = (float_pred.numpy()[mask] == cls_idx).sum() / mask.sum()
                print(f"  {cls_name:10s}: float={float_cls_acc:.3f}  i32={cls_acc:.3f}")

    return float_acc, float(fixed_acc)


def format_rust_array(arr: np.ndarray, name: str, shape_comment: str) -> str:
    """Format a numpy array as a Rust const slice."""
    flat = arr.flatten()
    values = ", ".join(str(v) for v in flat)

    # Split into rows for readability
    row_size = arr.shape[-1] if arr.ndim > 1 else min(28, len(flat))
    rows = []
    for i in range(0, len(flat), row_size):
        chunk = flat[i : i + row_size]
        row = ", ".join(str(v) for v in chunk)
        if arr.ndim > 1 and i // row_size < arr.shape[0]:
            rows.append(f"    // Neuron {i // row_size}\n    {row},")
        else:
            rows.append(f"    {row},")

    body = "\n".join(rows)
    return f"/// {shape_comment}\nconst {name}: &[i32] = &[\n{body}\n];\n"


def export_to_rust(
    model: SkillSafetyMLP,
    num_classes: int = 4,
    output_path: str = None,
    verbose: bool = True,
):
    """Export model weights as Rust const arrays.

    If output_path is None, prints to stdout.
    """
    state = model.state_dict()

    # Map PyTorch state_dict keys to Rust const names
    layers = [
        ("fc1.weight", "W1", "fc1.bias", "B1"),
        ("fc2.weight", "W2", "fc2.bias", "B2"),
        ("fc3.weight", "W3", "fc3.bias", "B3"),
    ]

    layer_descriptions = [
        ("Layer 1 weights", "[56 hidden neurons, 45 input features]"),
        ("Layer 2 weights", "[40 hidden neurons, 56 neurons from layer 1]"),
        ("Layer 3 (output) weights", f"[{num_classes} output neurons, 40 hidden neurons]"),
    ]

    output_lines = []
    output_lines.append("// Auto-generated weights from training script")
    output_lines.append(f"// Training seed: 42")
    output_lines.append(
        "// All weights use fixed-point arithmetic at scale=7 (multiplied by 2^7 = 128)."
    )
    output_lines.append(
        "// Input features are pre-normalized to [0, 128].\n"
    )

    for (w_key, w_name, b_key, b_name), (desc, shape) in zip(
        layers, layer_descriptions
    ):
        w = state[w_key].numpy()
        b = state[b_key].numpy()

        w_fixed = float_to_fixed(w)
        b_fixed = float_to_fixed(b)

        output_lines.append(format_rust_array(w_fixed, w_name, f"{desc}: {shape}"))
        output_lines.append(
            format_rust_array(b_fixed, b_name, f"Layer bias: [{b_fixed.shape[0]}]")
        )

    # Param count
    total_params = sum(p.numel() for p in model.parameters())
    output_lines.append(f"// Total parameters: {total_params}")

    rust_code = "\n".join(output_lines)

    if output_path:
        Path(output_path).write_text(rust_code)
        if verbose:
            print(f"Exported weights to {output_path}")
    else:
        print(rust_code)

    return rust_code


def main():
    parser = argparse.ArgumentParser(description="Export model weights to Rust")
    parser.add_argument(
        "--model", type=str, default="data/model.pt", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=4,
        help="Number of output classes (3 for real data, 4 for synthetic)"
    )
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "real"],
        help="Dataset type for validation"
    )
    parser.add_argument(
        "--real-labels", type=str, default="training/real-labels.json",
        help="Path to real labeled dataset (used with --dataset real)"
    )
    parser.add_argument("--validate", action="store_true", help="Run roundtrip validation")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    model = SkillSafetyMLP(input_dim=NUM_FEATURES, num_classes=args.num_classes, qat=False)
    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    if args.validate:
        if args.dataset == "real":
            features, labels = load_real_dataset(args.real_labels)
        else:
            features, labels = generate_dataset()
        float_acc, fixed_acc = validate_roundtrip(
            model, features, labels, num_classes=args.num_classes, verbose=verbose
        )
        if fixed_acc < 0.90:
            print(f"WARNING: Fixed-point accuracy ({fixed_acc:.4f}) below 90% threshold")
            return 1

    export_to_rust(model, num_classes=args.num_classes, output_path=args.output, verbose=verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
