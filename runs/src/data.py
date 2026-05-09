from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class Batch:
    input_ids: torch.Tensor           # (B, n)
    attention_mask: torch.Tensor      # (B, n)
    labels: torch.Tensor              # (B, m) multi-hot float32
    label_ids_list: List[List[int]]   # length B, each is list of positive label ids


class MLTCTextDataset(Dataset):
    """Reads a split csv with columns: Texts, NumberLabels (space-separated ids)."""

    def __init__(self, csv_path: str, num_labels: int):
        self.df = pd.read_csv(csv_path)
        if "Texts" not in self.df.columns or "NumberLabels" not in self.df.columns:
            raise ValueError(f"CSV must contain 'Texts' and 'NumberLabels': {csv_path}")
        self.texts = self.df["Texts"].astype(str).tolist()
        self.label_strs = self.df["NumberLabels"].astype(str).tolist()
        self.num_labels = num_labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, List[int]]:
        text = self.texts[idx]
        raw = self.label_strs[idx].strip()
        if raw == "" or raw.lower() == "nan":
            label_ids: List[int] = []
        else:
            label_ids = [int(x) for x in raw.split()]
        return text, label_ids


def make_collate_fn(tokenizer: PreTrainedTokenizerBase, num_labels: int, max_length: int):

    def collate(examples: List[Tuple[str, List[int]]]) -> Batch:
        texts, label_ids_list = zip(*examples)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        B = len(texts)
        y = torch.zeros((B, num_labels), dtype=torch.float32)

        for i, lab in enumerate(label_ids_list):
            for lid in lab:
                if lid < 0 or lid >= num_labels:
                    raise ValueError(f"Label id {lid} out of range 0..{num_labels-1}")
                y[i, lid] = 1.0

        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=y,
            label_ids_list=[list(l) for l in label_ids_list],
        )
    return collate


def build_label_description_inputs(
    tokenizer: PreTrainedTokenizerBase,
    index_to_label: List[str],
    label_desc: Dict[str, str],
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """Tokenize label descriptions once. Returns dict with input_ids/attention_mask (m, L)."""
    desc_texts = []
    for name in index_to_label:
        if name not in label_desc:
            raise ValueError(f"Missing description for label '{name}'")
        # prepend label name for extra signal (optional but harmless)
        desc_texts.append(f"{name}: {label_desc[name]}")
        
    enc = tokenizer(
        desc_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def compute_label_stats_from_ids(
    label_ids_list: List[List[int]],
    num_labels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freq, cooccur) from a list of label id lists (training split)."""
    freq = np.zeros((num_labels,), dtype=np.int64)
    co = np.zeros((num_labels, num_labels), dtype=np.int64)
    for labs in label_ids_list:
        if not labs:
            continue
        labs_uniq = sorted(set(labs))
        for i in labs_uniq:
            freq[i] += 1
        # document-level co-occurrence: count once per doc
        for i in range(len(labs_uniq)):
            a = labs_uniq[i]
            for j in range(i, len(labs_uniq)):
                b = labs_uniq[j]
                co[a, b] += 1
                if a != b:
                    co[b, a] += 1
    return freq, co
