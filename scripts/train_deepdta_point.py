#!/usr/bin/env python3
"""
DeepDTA point model training + prediction export compatible with existing CP pipeline.

Key goals:
- Paper-faithful DeepDTA-style CNN backbone (1D CNN + FC).
- No leakage: DO NOT use calibration split for early stopping; create a small val split from train.
- Export artifacts compatible with existing CP scripts:
  - preds_cal.csv.gz
  - preds_test.csv.gz
  - metrics.json
  - config.json

Important fix for Davis in this repo:
- Davis label `y` appears to be Kd in nM (capped at 10000). For standard DeepDTA/DTI regression,
  we usually use pKd scale. We support:
    pKd = 9 - log10(Kd_nM)
- Use --label_transform auto (default) to apply davis_pkd for Davis, and none for others.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Metrics
# -----------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cindex(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = 0
    h_sum = 0.0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            n += 1
            if (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]) or (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]):
                h_sum += 1.0
            elif y_pred[i] == y_pred[j]:
                h_sum += 0.5
    return float(h_sum / n) if n > 0 else float("nan")


def get_git_commit(path: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


# -----------------------------
# Label transform
# -----------------------------
def transform_y(y: np.ndarray, mode: str, dataset: str) -> np.ndarray:
    """
    Transform labels into a consistent scale.

    - none: keep as-is
    - log10: log10(y)
    - davis_pkd: pKd = 9 - log10(Kd_nM)
    - auto: davis -> davis_pkd, else none
    """
    mode = str(mode).lower()
    dataset = str(dataset).lower()
    y = np.asarray(y, dtype=np.float64)

    if mode == "auto":
        mode = "davis_pkd" if dataset == "davis" else "none"

    if mode == "none":
        return y

    y_safe = np.clip(y, 1e-12, None)

    if mode == "log10":
        return np.log10(y_safe)

    if mode == "davis_pkd":
        return 9.0 - np.log10(y_safe)

    raise ValueError(f"Unknown label_transform mode: {mode}")


# -----------------------------
# Split indices loader (robust to naming)
# -----------------------------
def load_split_indices(split_dir: Path) -> Dict[str, np.ndarray]:
    name_map = {
        "train": ["idx_train.npy", "train_idx.npy", "idx_tr.npy"],
        "cal": ["idx_cal.npy", "cal_idx.npy", "idx_valid.npy", "idx_val.npy", "valid_idx.npy", "val_idx.npy"],
        "test": ["idx_test.npy", "test_idx.npy"],
    }
    out: Dict[str, np.ndarray] = {}
    for key, candidates in name_map.items():
        found = None
        for fn in candidates:
            p = split_dir / fn
            if p.exists():
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Missing split index for '{key}' under {split_dir} (tried {candidates})")
        arr = np.load(found)
        out[key] = np.asarray(arr, dtype=np.int64)
    return out


# -----------------------------
# Encodings (DeepDTA-style)
# PAD=0, others are 1..K
# -----------------------------
CHARPROTSET: Dict[str, int] = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9,
    "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17,
    "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}

CHARISOSMISET: Dict[str, int] = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, "1": 35,
    "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7,
    "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, "D": 10, "G": 44, "F": 11,
    "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, "O": 48, "N": 14, "P": 15, "S": 49,
    "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54,
    "\\": 20, "a": 55, "c": 56, "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59,
    "h": 24, "m": 60, "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28,
    "y": 64,
}

PROT_VOCAB_SIZE = max(CHARPROTSET.values()) + 1
SMI_VOCAB_SIZE = max(CHARISOSMISET.values()) + 1


def encode_protein(seq: str, max_len: int) -> np.ndarray:
    seq = (seq or "").strip().upper()
    arr = np.zeros((max_len,), dtype=np.int64)
    n = min(len(seq), max_len)
    for i in range(n):
        arr[i] = int(CHARPROTSET.get(seq[i], 0))
    return arr


def encode_smiles(smi: str, max_len: int) -> np.ndarray:
    smi = (smi or "").strip()
    arr = np.zeros((max_len,), dtype=np.int64)
    n = min(len(smi), max_len)
    for i in range(n):
        arr[i] = int(CHARISOSMISET.get(smi[i], 0))
    return arr


# -----------------------------
# Token cache by entity id (speed)
# -----------------------------
@dataclass
class TokenCache:
    drug_tokens: Dict[int, np.ndarray]
    target_tokens: Dict[int, np.ndarray]


def build_token_cache(df_pairs: pd.DataFrame, max_smi_len: int, max_seq_len: int) -> TokenCache:
    drug_map = df_pairs.groupby("drug_idx")["smiles"].first().to_dict()
    target_map = df_pairs.groupby("target_idx")["sequence"].first().to_dict()

    drug_tokens: Dict[int, np.ndarray] = {}
    for k, smi in drug_map.items():
        drug_tokens[int(k)] = encode_smiles(str(smi), max_smi_len)

    target_tokens: Dict[int, np.ndarray] = {}
    for k, seq in target_map.items():
        target_tokens[int(k)] = encode_protein(str(seq), max_seq_len)

    return TokenCache(drug_tokens=drug_tokens, target_tokens=target_tokens)


class DeepDTAPairDataset(Dataset):
    def __init__(
        self,
        df_pairs: pd.DataFrame,
        pair_indices: np.ndarray,
        y_col: str,
        cache: TokenCache,
        dataset: str,
        label_transform: str,
        save_text_cols: bool = False,
    ):
        self.df = df_pairs
        self.idxs = np.asarray(pair_indices, dtype=np.int64)
        self.y_col = y_col
        self.cache = cache
        self.dataset = dataset
        self.label_transform = label_transform
        self.save_text_cols = save_text_cols

        sub = self.df.iloc[self.idxs]
        self.drug_idx = sub["drug_idx"].to_numpy(dtype=np.int64)
        self.target_idx = sub["target_idx"].to_numpy(dtype=np.int64)

        y_raw = sub[y_col].to_numpy(dtype=np.float64)
        y_tr = transform_y(y_raw, mode=label_transform, dataset=dataset)
        self.y = y_tr.astype(np.float32)

        if save_text_cols:
            self.smiles = sub["smiles"].astype(str).tolist()
            self.sequence = sub["sequence"].astype(str).tolist()
        else:
            self.smiles = None
            self.sequence = None

    def __len__(self) -> int:
        return int(self.idxs.shape[0])

    def __getitem__(self, i: int):
        d = int(self.drug_idx[i])
        t = int(self.target_idx[i])
        x_smi = self.cache.drug_tokens.get(d, None)
        x_seq = self.cache.target_tokens.get(t, None)
        if x_smi is None:
            x_smi = np.zeros((100,), dtype=np.int64)
        if x_seq is None:
            x_seq = np.zeros((1000,), dtype=np.int64)

        item = {
            "x_smi": torch.from_numpy(x_smi).long(),
            "x_seq": torch.from_numpy(x_seq).long(),
            "y": torch.tensor(float(self.y[i]), dtype=torch.float32),
            "drug_idx": d,
            "target_idx": t,
            "pair_idx": int(self.idxs[i]),
        }
        if self.save_text_cols:
            item["smiles"] = self.smiles[i]
            item["sequence"] = self.sequence[i]
        return item


def collate_fn(batch: List[dict]) -> dict:
    x_smi = torch.stack([b["x_smi"] for b in batch], dim=0)
    x_seq = torch.stack([b["x_seq"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    out = {
        "x_smi": x_smi,
        "x_seq": x_seq,
        "y": y,
        "drug_idx": np.array([b["drug_idx"] for b in batch], dtype=np.int64),
        "target_idx": np.array([b["target_idx"] for b in batch], dtype=np.int64),
        "pair_idx": np.array([b["pair_idx"] for b in batch], dtype=np.int64),
    }
    if "smiles" in batch[0]:
        out["smiles"] = [b["smiles"] for b in batch]
        out["sequence"] = [b["sequence"] for b in batch]
    return out


# -----------------------------
# Model: DeepDTA-style CNN
# -----------------------------
class CNNBlock1D(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, kernel_size: int, num_filters: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(emb_dim, num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.emb(x)               # (B, L, E)
        z = z.transpose(1, 2)         # (B, E, L)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = torch.max(z, dim=2).values  # global max pool
        z = self.dropout(z)
        return z


class DeepDTA(nn.Module):
    def __init__(
        self,
        smi_vocab_size: int,
        prot_vocab_size: int,
        emb_dim: int = 128,
        smi_kernel: int = 8,
        prot_kernel: int = 12,
        num_filters: int = 32,
        fc_dims: Tuple[int, int, int] = (1024, 1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.smi_block = CNNBlock1D(smi_vocab_size, emb_dim, smi_kernel, num_filters, dropout)
        self.prot_block = CNNBlock1D(prot_vocab_size, emb_dim, prot_kernel, num_filters, dropout)

        in_dim = (num_filters * 3) + (num_filters * 3)
        d1, d2, d3 = fc_dims
        self.fc1 = nn.Linear(in_dim, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, d3)
        self.out = nn.Linear(d3, 1)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x_smi: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        a = self.smi_block(x_smi)
        b = self.prot_block(x_seq)
        z = torch.cat([a, b], dim=1)
        z = self.drop(F.relu(self.fc1(z)))
        z = self.drop(F.relu(self.fc2(z)))
        z = self.drop(F.relu(self.fc3(z)))
        y = self.out(z).squeeze(-1)
        return y


# -----------------------------
# Train / predict
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, dict]:
    model.eval()
    ys: List[float] = []
    ps: List[float] = []
    meta = {"drug_idx": [], "target_idx": [], "pair_idx": [], "smiles": [], "sequence": []}

    for batch in loader:
        x_smi = batch["x_smi"].to(device, non_blocking=True)
        x_seq = batch["x_seq"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        p = model(x_smi, x_seq)

        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(p.detach().cpu().numpy().tolist())
        meta["drug_idx"].extend(batch["drug_idx"].tolist())
        meta["target_idx"].extend(batch["target_idx"].tolist())
        meta["pair_idx"].extend(batch["pair_idx"].tolist())

        if "smiles" in batch:
            meta["smiles"].extend(batch["smiles"])
            meta["sequence"].extend(batch["sequence"])

    return np.asarray(ys, dtype=np.float64), np.asarray(ps, dtype=np.float64), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--max_seq_len", type=int, default=1000)
    ap.add_argument("--max_smi_len", type=int, default=100)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--num_filters", type=int, default=32)
    ap.add_argument("--smi_kernel", type=int, default=8)
    ap.add_argument("--prot_kernel", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--label_transform", type=str, default="auto",
                    choices=["auto", "none", "davis_pkd", "log10"],
                    help="Label transform applied to y before training and saved preds. Default: auto.")

    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (saves GPU memory).")
    ap.add_argument("--save_text_cols", action="store_true", help="Also save smiles/sequence in preds (bigger files).")
    ap.add_argument("--out_subdir", type=str, default="", help="If set, write outputs to this directory.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    set_seed(int(args.seed))

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pairs_path = repo / "data" / "processed" / args.dataset / "pairs.csv.gz"
    split_dir = repo / "data" / "processed" / args.dataset / "splits" / args.split / f"seed_{args.seed}"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing {pairs_path}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing {split_dir}")

    df_pairs = pd.read_csv(pairs_path, compression="gzip")

    # Infer label column name
    y_col = None
    for cand in ["y", "affinity", "label", "Y"]:
        if cand in df_pairs.columns:
            y_col = cand
            break
    if y_col is None:
        raise ValueError("Cannot infer label column from pairs.csv.gz (expected one of y/affinity/label/Y)")

    # Validate required columns
    for c in ["drug_idx", "target_idx", "smiles", "sequence"]:
        if c not in df_pairs.columns:
            raise ValueError(f"Missing required column '{c}' in pairs.csv.gz")

    idx = load_split_indices(split_dir)
    train_idx = idx["train"]
    cal_idx = idx["cal"]
    test_idx = idx["test"]

    # Train/Val split from train only (no leakage)
    rs = np.random.RandomState(int(args.seed))
    perm = train_idx.copy()
    rs.shuffle(perm)
    n_val = int(round(float(args.val_frac) * len(perm)))
    n_val = max(1, min(n_val, len(perm) - 1))
    val_idx = perm[:n_val]
    train_fit_idx = perm[n_val:]

    cache = build_token_cache(df_pairs, max_smi_len=int(args.max_smi_len), max_seq_len=int(args.max_seq_len))

    ds_train = DeepDTAPairDataset(df_pairs, train_fit_idx, y_col, cache,
                                  dataset=args.dataset, label_transform=args.label_transform, save_text_cols=False)
    ds_val = DeepDTAPairDataset(df_pairs, val_idx, y_col, cache,
                                dataset=args.dataset, label_transform=args.label_transform, save_text_cols=False)
    ds_cal = DeepDTAPairDataset(df_pairs, cal_idx, y_col, cache,
                                dataset=args.dataset, label_transform=args.label_transform, save_text_cols=bool(args.save_text_cols))
    ds_test = DeepDTAPairDataset(df_pairs, test_idx, y_col, cache,
                                 dataset=args.dataset, label_transform=args.label_transform, save_text_cols=bool(args.save_text_cols))

    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True,
                          num_workers=int(args.num_workers), pin_memory=True,
                          collate_fn=collate_fn, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False,
                        num_workers=int(args.num_workers), pin_memory=True,
                        collate_fn=collate_fn, drop_last=False)
    dl_cal = DataLoader(ds_cal, batch_size=int(args.batch_size), shuffle=False,
                        num_workers=int(args.num_workers), pin_memory=True,
                        collate_fn=collate_fn, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False,
                         num_workers=int(args.num_workers), pin_memory=True,
                         collate_fn=collate_fn, drop_last=False)

    model = DeepDTA(
        smi_vocab_size=SMI_VOCAB_SIZE + 2,
        prot_vocab_size=PROT_VOCAB_SIZE + 2,
        emb_dim=int(args.emb_dim),
        smi_kernel=int(args.smi_kernel),
        prot_kernel=int(args.prot_kernel),
        num_filters=int(args.num_filters),
        fc_dims=(1024, 1024, 512),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # AMP compatibility across torch versions
    try:
        from torch.amp import GradScaler, autocast  # type: ignore
        scaler = GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")
        autocast_ctx = lambda: autocast("cuda", enabled=scaler.is_enabled())
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=scaler.is_enabled())

    # Output dir
    if args.out_subdir.strip():
        out_dir = Path(args.out_subdir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y-%m-%d")
        exp = f"deepdta_point_default_{args.dataset}_{args.split}_seed{args.seed}"
        out_dir = repo / "runs" / stamp / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    ckpt_path = out_dir / "model.pt"

    for epoch in range(int(args.max_epochs)):
        model.train()
        losses: List[float] = []

        for batch in dl_train:
            x_smi = batch["x_smi"].to(device, non_blocking=True)
            x_seq = batch["x_seq"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                p = model(x_smi, x_seq)
                loss = F.mse_loss(p, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu().item()))

        yv, pv, _ = predict(model, dl_val, device)
        val_rmse = rmse(yv, pv)

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path)
        else:
            bad_epochs += 1

        print(json.dumps({
            "epoch": epoch,
            "train_mse": float(np.mean(losses)) if losses else None,
            "val_rmse": val_rmse,
            "best_val_rmse": best_val,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs
        }, indent=2), flush=True)

        if bad_epochs >= int(args.patience):
            break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    y_cal, p_cal, meta_cal = predict(model, dl_cal, device)
    y_test, p_test, meta_test = predict(model, dl_test, device)

    def save_preds(path: Path, y: np.ndarray, p: np.ndarray, meta: dict) -> None:
        out = pd.DataFrame({
            "y": y.astype(np.float64),
            "y_pred": p.astype(np.float64),
            "yhat": p.astype(np.float64),  # alias for compatibility
            "drug_idx": np.asarray(meta["drug_idx"], dtype=np.int64),
            "target_idx": np.asarray(meta["target_idx"], dtype=np.int64),
            "pair_idx": np.asarray(meta["pair_idx"], dtype=np.int64),
        })
        if meta.get("smiles"):
            out["smiles"] = meta["smiles"]
            out["sequence"] = meta["sequence"]
        out.to_csv(path, index=False, compression="gzip")

    save_preds(out_dir / "preds_cal.csv.gz", y_cal, p_cal, meta_cal)
    save_preds(out_dir / "preds_test.csv.gz", y_test, p_test, meta_test)

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": int(args.seed),
        "model": "deepdta",
        "label_transform": args.label_transform,
        "code_commit": get_git_commit(repo),
        "out_dir": str(out_dir),
        "n_train": int(len(train_fit_idx)),
        "n_val": int(len(val_idx)),
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val),

        "rmse_cal": rmse(y_cal, p_cal),
        "rmse_test": rmse(y_test, p_test),
        "cindex_cal": cindex(y_cal, p_cal),
        "cindex_test": cindex(y_test, p_test),

        # aliases for older collectors
        "rmse_cal_point": rmse(y_cal, p_cal),
        "rmse_test_point": rmse(y_test, p_test),
        "cindex_cal_point": cindex(y_cal, p_cal),
        "cindex_test_point": cindex(y_test, p_test),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    config = vars(args)
    config.update({
        "pairs_path": str(pairs_path),
        "split_dir": str(split_dir),
        "train_fit_idx_n": int(len(train_fit_idx)),
        "val_idx_n": int(len(val_idx)),
        "cal_idx_n": int(len(cal_idx)),
        "test_idx_n": int(len(test_idx)),
        "device_used": str(device),
    })
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[DONE] out_dir={out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
