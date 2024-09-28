"""The positional encoding layer for the transformer architecture."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """The positional encoding from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf)."""

    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1) -> None:
        """Initialize a positional encoding layer.

        Args:
            d_model: the model's embedding dimension (default=512).
            max_len: the max number of positions (default=5000).
            dropout: the dropout value (default=0.1).
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def _compute_positions(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies positional encoding to the input tensor.

        Args:
            x: the input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            a tensor on which sinusoidal positional encoding was applied.
        """
        x += self._compute_positions(x)
        return self.dropout(x)
