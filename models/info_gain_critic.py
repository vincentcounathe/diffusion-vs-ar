"""Information-gain critic implementation.

This module defines a lightweight MLP (g_\\phi) that operates on per-token
feature vectors produced by :mod:`utils.feature_extractor`.  It predicts the
counterfactual reduction in cross-entropy (\\Delta CE) when the ground-truth
value of a token is revealed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class CriticConfig:
    """Configuration container for :class:`InfoGainCritic`."""

    input_dim: int
    hidden_dims: Sequence[int] = (256, 128)
    dropout: float = 0.1
    activation: str = "gelu"

    def activation_layer(self) -> nn.Module:
        if self.activation.lower() == "relu":
            return nn.ReLU()
        if self.activation.lower() == "gelu":
            return nn.GELU()
        if self.activation.lower() == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {self.activation}")


class InfoGainCritic(nn.Module):
    """A small MLP that predicts token-level information gain."""

    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        activation = config.activation_layer()

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict \\Delta CE for every feature vector.

        Args:
            features: Tensor of shape ``(..., input_dim)`` containing per-token
                feature vectors.

        Returns:
            Tensor of shape ``(...,)`` with the predicted \\Delta CE values.
        """

        pred = self.mlp(features)
        return pred.squeeze(-1)

    @classmethod
    def from_dims(
        cls,
        input_dim: int,
        hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> "InfoGainCritic":
        """Create a critic directly from dimension arguments."""

        config = CriticConfig(
            input_dim=input_dim,
            hidden_dims=tuple(hidden_dims or (256, 128)),
            dropout=dropout,
            activation=activation,
        )
        return cls(config)
