from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Return torch.device.

    If `device` is None, selects CUDA when available, else CPU.
    """
    if device is not None and device.strip():
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
