"""
PyTorch MLP matching the Rust SkillGuard architecture.

Architecture: 35 -> 56 -> 40 -> 4 (4,460 params)
Fixed-point scale=7 (multiply by 128).
"""

import torch
import torch.nn as nn


class FixedPointLinear(nn.Module):
    """Linear layer that simulates fixed-point quantization during forward pass.

    Uses straight-through estimator for gradient flow through quantization.
    This ensures trained weights are robust to i32 conversion.
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
        # Quantize weights: round(w * scale) / scale (straight-through estimator)
        w_q = self.weight + (
            torch.round(self.weight * self.scale) / self.scale - self.weight
        ).detach()
        b_q = self.bias + (
            torch.round(self.bias * self.scale) / self.scale - self.bias
        ).detach()
        return nn.functional.linear(x, w_q, b_q)


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
