"""
MinimalTST — минималистичная модель Time Series Transformer для регрессии.
- 8 энкодер-слоёв, d_model=128, nhead=8.
- Вызов: model = MinimalTST(); output = model(x), где x — [batch, seq_len, 1].
"""

import torch
import torch.nn as nn

class MinimalTST(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=8, seq_len=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        self.seq_len = seq_len

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.encoder(x)
        out = self.head(x[:, -1, :])
        return out.squeeze(-1)
