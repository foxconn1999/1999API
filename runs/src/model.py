from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .dgcn import DGCN


def _row_normalize_torch(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return A / (A.sum(dim=1, keepdim=True) + eps)


class MLTCMedoidCLModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        ppmi_adj: torch.Tensor,   # (m, m) float32
        dgcn_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        if ppmi_adj.shape != (num_labels, num_labels):
            raise ValueError(f"ppmi_adj shape must be ({num_labels},{num_labels}), got {ppmi_adj.shape}")
        self.register_buffer("ppmi_adj", ppmi_adj.float())

        self.dgcn = DGCN(hidden_size=hidden, num_layers=dgcn_layers, dropout=dropout)
        self.classifier = nn.Linear(hidden, 1)  # shared head
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        doc_input_ids: torch.Tensor,        # (B, n)
        doc_attention_mask: torch.Tensor,   # (B, n)
        label_input_ids: torch.Tensor,      # (m, L)
        label_attention_mask: torch.Tensor, # (m, L)
        rho: float,                         # scalar in [0,1]
    ) -> Dict[str, torch.Tensor]:
        device = doc_input_ids.device
        m = self.num_labels

        # ---- Encode label descriptions (C0) ----
        label_out = self.bert(
            input_ids=label_input_ids.to(device),
            attention_mask=label_attention_mask.to(device),
        )
        C0 = label_out.last_hidden_state[:, 0, :]  # (m, d)

        d = C0.size(-1)
        # Dynamic label-label attention adjacency (row-softmax)
        att_scores = torch.matmul(C0, C0.T) / (d ** 0.5)  # (m, m)
        A_att = torch.softmax(att_scores, dim=1)          # rows sum 1

        # PPMI row-normalize (cached)
        A_ppmi = _row_normalize_torch(self.ppmi_adj)

        # Mix with rho(t) and re-normalize
        rho_t = float(rho)
        A_mix = rho_t * A_ppmi + (1.0 - rho_t) * A_att
        A_mix = _row_normalize_torch(A_mix)

        # ---- DGCN ----
        C = self.dgcn(A_mix, C0)  # (m, d)

        # ---- Encode document tokens (Z) ----
        doc_out = self.bert(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
        )
        Z = doc_out.last_hidden_state  # (B, n, d)
        Z = self.dropout(Z)

        # ---- beta-row (B,m,n) ----
        beta_row = torch.einsum("md,bnd->bmn", C, Z) / (d ** 0.5)

        # ---- beta-soft with padding mask ----
        token_mask = doc_attention_mask.unsqueeze(1).bool()  # (B,1,n)
        beta_for_soft = beta_row.masked_fill(~token_mask, -1e9)
        beta_soft = torch.softmax(beta_for_soft, dim=-1)  # (B,m,n)

        # ---- z~ (B,m,d) ----
        z_tilde = torch.einsum("bmn,bnd->bmd", beta_soft, Z)

        # ---- beta_row_sum (B,m): sum ReLU(beta_row) over valid tokens ----
        valid = token_mask.float()
        relu_beta_valid = torch.relu(beta_row) * valid                          # (B,m,n)
        valid_token_count = valid.sum(dim=-1).clamp(min=1.0)       # (B,1)
        beta_row_mean = relu_beta_valid.sum(dim=-1) / valid_token_count  # (B,m)
        s_c = torch.sigmoid(beta_row_mean)

        # beta_row_sum = (torch.relu(beta_row) * valid).sum(dim=-1)  # (B,m)

        # ---- shared classifier ----
        logits = self.classifier(z_tilde).squeeze(-1)  # (B,m)

        return {
            "logits": logits,
            "z_tilde": z_tilde,
            "beta_row": beta_row,
            "s_c": s_c,
        }
