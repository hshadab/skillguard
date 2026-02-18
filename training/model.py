"""
PyTorch MLP matching the Rust SkillGuard architecture.

Architecture: 35 -> 56 -> 40 -> 4 (4,460 params)
Fixed-point scale=7 (multiply by 2^7 = 128).

QAT simulates the exact Rust i32 inference path:
  matmul(x_i32, w_i32) → (result + 64) // 128 + bias_i32 → relu → repeat
This forces the model to learn weight configurations that produce correct
classifications even after integer truncation at each layer.
"""

import torch
import torch.nn as nn


class FixedPointLinear(nn.Module):
    """Linear layer that simulates the full Rust fixed-point inference path.

    Rust inference for each layer:
      1. mm = matmul(input_i32, weight_i32)   — both at scale=128
      2. mm_rounded = mm + 64                  — rounding bias
      3. mm_rescaled = mm_rounded / 128        — integer division (floor)
      4. biased = mm_rescaled + bias_i32       — add bias at scale=128
      5. output = relu(biased)

    QAT simulates this using straight-through estimators (STE):
    the forward pass computes the quantized result, but gradients flow
    through as if the operations were identity functions.
    """

    def __init__(self, in_features: int, out_features: int, scale: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate Rust i32 inference with straight-through gradient estimation.

        The entire forward pass operates in "i32 integer units".
        Input x should be integer-valued (features in [0, 128] or i32 from prev layer).
        Output is integer-valued (i32 logits or activations for next layer).

        Rust path per layer:
          mm = x_i32 @ W_i32.T             (integer matmul)
          rescaled = (mm + 64) // 128       (integer floor-division)
          output = rescaled + B_i32         (integer bias add)

        STE: gradients flow through round/floor as identity.
        """
        S = self.scale  # 128

        # Get i32 weights and bias via STE
        # Forward: round(w * S), Backward: w * S (identity through round)
        w_i32 = self.weight * S + (torch.round(self.weight * S) - self.weight * S).detach()
        b_i32 = self.bias * S + (torch.round(self.bias * S) - self.bias * S).detach()

        # Quantize input to integers via STE
        x_int = x + (torch.round(x) - x).detach()

        # Integer matmul: x_int @ w_i32.T
        mm = nn.functional.linear(x_int, w_i32, None)

        # Integer floor division: (mm + 64) // 128
        # STE: forward = floor((mm+64)/128), backward = mm/128
        half = S // 2  # 64
        divided_exact = (mm + half) / S  # smooth version for gradient
        divided_floor = torch.floor(divided_exact)  # integer floor
        divided = divided_exact + (divided_floor - divided_exact).detach()

        # Add i32 bias
        result = divided + b_i32

        return result


class SkillSafetyMLP(nn.Module):
    """3-layer MLP for skill safety classification.

    Architecture: input_dim -> 56 -> 40 -> num_classes
    Activation: ReLU between hidden layers, no activation on output.
    """

    def __init__(self, input_dim: int = 35, num_classes: int = 4, qat: bool = True):
        super().__init__()
        self.qat = qat
        self.num_classes = num_classes

        if qat:
            self.fc1 = FixedPointLinear(input_dim, 56)
            self.fc2 = FixedPointLinear(56, 40)
            self.fc3 = FixedPointLinear(40, num_classes)
        else:
            self.fc1 = nn.Linear(input_dim, 56)
            self.fc2 = nn.Linear(56, 40)
            self.fc3 = nn.Linear(40, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output (raw logits)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=4)
    args = parser.parse_args()

    nc = args.num_classes
    model = SkillSafetyMLP(input_dim=35, num_classes=nc, qat=True)
    print(f"Architecture: 35 -> 56 -> 40 -> {nc}")
    print(f"Total parameters: {model.param_count()}")
    print(f"  Layer 1: {35 * 56 + 56} = {35 * 56 + 56}")
    print(f"  Layer 2: {56 * 40 + 40} = {56 * 40 + 40}")
    print(f"  Layer 3: {40 * nc + nc} = {40 * nc + nc}")
    print(f"  Total: {35 * 56 + 56 + 56 * 40 + 40 + 40 * nc + nc}")

    # Test forward pass
    x = torch.randn(1, 35)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")
