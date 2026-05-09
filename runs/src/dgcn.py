from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCN(nn.Module):
    """A minimal multi-layer graph convolution for label embeddings.

    Each layer performs:
        H <- ReLU( A @ (W H) )
    where A is a row-normalized adjacency (already mixed).
    """

    def __init__(self, hidden_size: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, A: torch.Tensor, C0: torch.Tensor) -> torch.Tensor:
        """Args:
        A: (m, m) row-normalized adjacency (already mixed).
        C0: (m, d) input label embeddings.
        """
        H = C0
        for layer in range(self.num_layers):
            H = self.proj[layer](H)
            H = self.dropout(H)
            H = torch.matmul(A, H)
            H = F.relu(H)
            
        return H
