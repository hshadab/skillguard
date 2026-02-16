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

from dataset import CLASS_NAMES, NUM_FEATURES, generate_dataset, SkillDataset
from model import SkillSafetyMLP


SCALE = 128  # 2^7


def float_to_fixed(weight: np.ndarray) -> np.ndarray:
    """Convert float32 weights to i32 fixed-point (scale=128)."""
    return np.round(weight * SCALE).astype(np.int32)


def fixed_to_float(fixed: np.ndarray) -> np.ndarray:
    """Convert i32 fixed-point back to float32."""
    return fixed.astype(np.float32) / SCALE


def validate_roundtrip(
    model: SkillSafetyMLP,
    features: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True,
) -> tuple[float, float]:
    """Validate that fixed-point conversion preserves classification accuracy.

    Simulates the Rust fixed-point inference path:
    1. Quantize weights to i32 (round(w * 128))
    2. Run inference with quantized weights
    3. Compare classifications

    Returns (float_accuracy, fixed_accuracy).
    """
    model.eval()

    # Float inference
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        float_output = model(x)
        _, float_pred = float_output.max(1)
        float_acc = float_pred.eq(y).sum().item() / len(y)

    # Fixed-point inference simulation
    state = model.state_dict()
    fixed_state = {}
    for key, val in state.items():
        w_np = val.numpy()
        w_fixed = float_to_fixed(w_np)
        w_back = fixed_to_float(w_fixed)
        fixed_state[key] = torch.tensor(w_back, dtype=torch.float32)

    # Create a copy of the model with fixed-point weights
    fixed_model = SkillSafetyMLP(input_dim=NUM_FEATURES, qat=False)
    fixed_model.load_state_dict(fixed_state)
    fixed_model.eval()

    with torch.no_grad():
        fixed_output = fixed_model(x)
        _, fixed_pred = fixed_output.max(1)
        fixed_acc = fixed_pred.eq(y).sum().item() / len(y)

    # Decision match (SAFE/CAUTION -> allow, DANGEROUS/MALICIOUS -> deny)
    float_decisions = (float_pred >= 2).long()
    fixed_decisions = (fixed_pred >= 2).long()
    decision_match = float_decisions.eq(fixed_decisions).sum().item() / len(y)

    if verbose:
        print(f"\n=== Roundtrip Validation ===")
        print(f"Float accuracy:     {float_acc:.4f}")
        print(f"Fixed-point accuracy: {fixed_acc:.4f}")
        print(f"Decision match:     {decision_match:.4f}")
        print(f"Classification diff: {(float_pred != fixed_pred).sum().item()} samples")

    return float_acc, fixed_acc


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
        ("Layer 1 weights", "[56 hidden neurons, 35 input features]"),
        ("Layer 2 weights", "[40 hidden neurons, 56 neurons from layer 1]"),
        ("Layer 3 (output) weights", "[4 output neurons, 40 hidden neurons]"),
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
    parser.add_argument("--validate", action="store_true", help="Run roundtrip validation")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    model = SkillSafetyMLP(input_dim=NUM_FEATURES, qat=False)
    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    if args.validate:
        features, labels = generate_dataset()
        float_acc, fixed_acc = validate_roundtrip(model, features, labels, verbose=verbose)
        if fixed_acc < 0.90:
            print(f"WARNING: Fixed-point accuracy ({fixed_acc:.4f}) below 90% threshold")
            return 1

    export_to_rust(model, output_path=args.output, verbose=verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
