"""The MTST classifier.

A small Transformer encoder over a window of market features, emitting logits
over {SHORT, HOLD, LONG}.

The defaults are deliberately modest — ``d_model=64``, 2 layers, 4 heads,
roughly 10^5 parameters. The original 8-layer/128-dim/8-head stack had on the
order of 10^6 parameters, which for a few thousand hourly candles is several
parameters per training sample. Model capacity should be justified by data
volume and by beating the baselines in ``nn/baselines.py``, not chosen for
impressiveness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from chimera.contracts import CLASS_ORDER


@dataclass
class MTSTConfig:
    """Architecture hyperparameters."""

    input_dim: int
    seq_len: int
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    n_classes: int = len(CLASS_ORDER)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.input_dim < 1 or self.seq_len < 1:
            raise ValueError("input_dim and seq_len must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MTSTConfig":
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})


class MTST(nn.Module):
    """Multivariate time-series Transformer classifier.

    Input ``(batch, seq_len, input_dim)`` -> logits ``(batch, n_classes)``.
    """

    def __init__(self, config: MTSTConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        # Learned positional embedding: the sequence length is fixed by the
        # metadata contract, so there is no need for the sinusoidal variant.
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.seq_len, config.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor has no effect with norm_first=True and only
        # emits a warning; turn it off explicitly.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq_len, features), got {tuple(x.shape)}")
        if x.shape[1] != self.config.seq_len:
            raise ValueError(
                f"expected sequence length {self.config.seq_len}, got {x.shape[1]}"
            )
        if x.shape[2] != self.config.input_dim:
            raise ValueError(f"expected {self.config.input_dim} features, got {x.shape[2]}")
        h = self.input_proj(x) + self.pos_embedding
        h = self.encoder(h)
        # Classify from the most recent step: it is the one the decision is
        # made at, and mean-pooling would blur it with older context.
        return self.head(h[:, -1])
