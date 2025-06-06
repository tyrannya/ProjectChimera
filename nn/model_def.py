import torch
import torch.nn as nn

class MTST(nn.Module):
    def __init__(self, input_dim: int = 1, seq_len: int = 100):
        super().__init__()
        d_model = 128
        nhead = 8
        num_layers = 8
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)
