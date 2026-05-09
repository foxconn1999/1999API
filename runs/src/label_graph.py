from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def compute_ppmi_adjacency(
    freq: np.ndarray,
    cooccur: np.ndarray,
    num_docs: int,
    eta: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute PPMI adjacency matrix (m, m) from document-level co-occurrence.

    Args:
        freq: shape (m,), document frequency per label (>=1 as per your data assumption).
        cooccur: shape (m, m), co-occurrence counts (docs containing both labels).
        num_docs: number of documents in the split used for building the graph (train).
        eta: frequency suppression hyperparameter.
    Returns:
        A_ppmi: shape (m, m), with diagonal set to 1.
    """
    m = int(freq.shape[0])
    N = float(num_docs)
    if N <= 0:
        raise ValueError("num_docs must be positive.")

    p_i = freq.astype(np.float64) / (N + eps)
    p_ij = cooccur.astype(np.float64) / (N + eps)

    omega = np.power(np.log1p(freq.astype(np.float64) + eps), eta)

    A = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            #denom = (p_i[i] + eps) * (p_i[j] + eps) + eps
            denom = (np.power(p_i[i] + eps, omega[i]) * np.power(p_i[j] + eps, omega[j])) + eps
            val = np.log((p_ij[i, j] + eps) / denom)
            if val > 0:
                A[i, j] = float(val)
                
    np.fill_diagonal(A, 1.0)
    return A


def row_normalize(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sum = A.sum(axis=1, keepdims=True)
    return A / (row_sum + eps)


def maybe_load_ppmi(cache_path: str) -> Tuple[bool, np.ndarray]:
    if os.path.exists(cache_path):
        return True, np.load(cache_path)
    return False, np.array([])


def save_ppmi(cache_path: str, A: np.ndarray) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, A)
