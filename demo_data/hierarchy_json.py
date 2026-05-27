from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def load_hierarchy_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    return records


def _hash_int(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _tokens_for_record(record: dict[str, Any]) -> list[str]:
    parent = str(record.get("parent", ""))
    level = str(record.get("level", ""))
    concept = str(record.get("concept", ""))
    sentence = str(record.get("sentence", ""))
    text = f"parent_{parent} level_{level} concept_{concept} {sentence}".lower()
    return _WORD_RE.findall(text)


def hierarchy_records_to_tensors(
    records: list[dict[str, Any]],
    *,
    feature_dim: int = 128,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """
    Convert hierarchy text records into deterministic hashed bag-of-words features.

    Returns:
      features: (N, feature_dim) float tensor
      level_ids: (N,) long tensor, useful as optional token/metadata IDs
      level_to_id: mapping from string level to integer ID
    """
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")

    levels = sorted({str(r.get("level", "")) for r in records})
    level_to_id = {level: idx for idx, level in enumerate(levels)}

    features = torch.zeros((len(records), feature_dim), dtype=dtype)
    level_ids = torch.empty((len(records),), dtype=torch.long)

    for row_idx, record in enumerate(records):
        level_ids[row_idx] = level_to_id[str(record.get("level", ""))]
        for token in _tokens_for_record(record):
            h = _hash_int(token)
            col = h % feature_dim
            sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
            features[row_idx, col] += sign

    norms = features.norm(dim=1, keepdim=True).clamp_min(1e-6)
    features = features / norms
    return features, level_ids, level_to_id


def make_hierarchy_dataloaders(
    path: str | Path,
    *,
    feature_dim: int = 128,
    batch_size: int = 256,
    val_fraction: float = 0.2,
    max_examples: int | None = 2_000,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Load the bundled hierarchy JSON and return train/validation DataLoaders.

    Batches are shaped as (features, level_ids), which the MFA pipeline accepts.
    The level IDs are optional metadata; projected K-Means only uses them when
    USE_TOKEN_WEIGHTS=True.
    """
    records = load_hierarchy_records(path)
    if max_examples is not None:
        records = records[: int(max_examples)]

    features, level_ids, level_to_id = hierarchy_records_to_tensors(
        records,
        feature_dim=feature_dim,
        dtype=dtype,
    )

    n = features.shape[0]
    if n == 0:
        raise ValueError("No hierarchy records available after max_examples filtering.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    gen = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(n, generator=gen)
    val_n = int(round(n * val_fraction))
    val_idx = order[:val_n]
    train_idx = order[val_n:]

    train_ds = TensorDataset(features[train_idx], level_ids[train_idx])
    val_ds = TensorDataset(features[val_idx], level_ids[val_idx]) if val_n else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, level_to_id
