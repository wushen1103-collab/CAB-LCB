#!/usr/bin/env python3
import argparse
import gzip
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

try:
    from torch_geometric.loader import DataLoader
except Exception:
    from torch_geometric.data import DataLoader
from torch_geometric.data import Data as GeoData


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_graphdta_dir(root: Path) -> Path:
    candidates = [
        root / "GraphDTA",
        root / "graphdta",
        root / "data" / "raw" / "GraphDTA",
        root / "data" / "raw" / "graphdta",
        root / "third_party" / "GraphDTA",
        root / "third_party" / "graphdta",
        root / "data" / "raw" / "graphdta",
        root / "data" / "raw" / "graphdta" / "GraphDTA",
        root / "data" / "raw" / "graphdta" / "graphdta",
    ]
    for p in candidates:
        if (p / "models").exists():
            return p
    msg = ["Cannot locate GraphDTA directory (need models/). Tried:"]
    msg.extend([f" - {str(p)}" for p in candidates])
    raise FileNotFoundError("\n".join(msg))


def get_git_commit(path: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


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
            raise FileNotFoundError(f"Cannot find {key} indices in {split_dir}")
        out[key] = np.asarray(np.load(found), dtype=np.int64)
    return out


def _normalize_colname(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _pick_column(df: pd.DataFrame, preferred: List[str], contains_tokens: List[str]) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in cols}

    for c in preferred:
        cc = norm_map.get(_normalize_colname(c))
        if cc is not None:
            return cc

    for c in cols:
        nc = _normalize_colname(c)
        for tok in contains_tokens:
            if tok in nc:
                return c
    return None


def infer_pair_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    smiles_col = _pick_column(
        df,
        preferred=["smiles", "canonical_smiles", "drug_smiles", "ligand_smiles", "compound_smiles"],
        contains_tokens=["smiles"],
    )
    seq_col = _pick_column(
        df,
        preferred=["sequence", "protein_sequence", "target_sequence", "protein", "target", "seq"],
        contains_tokens=["sequence", "protein", "target", "seq"],
    )
    if smiles_col is None or seq_col is None:
        raise KeyError(f"Cannot infer SMILES/sequence columns. columns={list(df.columns)}")

    y_col = _pick_column(
        df,
        preferred=["affinity", "y", "label", "score", "pki", "pkd", "pic50", "log_kd", "log_ki", "logki"],
        contains_tokens=["affinity", "label", "score", "pic50", "pkd", "pki", "log"],
    )
    if y_col is not None and pd.api.types.is_numeric_dtype(df[y_col]):
        return smiles_col, seq_col, y_col

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    blocked = {smiles_col, seq_col}
    candidates: List[str] = []
    for c in numeric_cols:
        if c in blocked:
            continue
        nc = _normalize_colname(c)
        if nc.endswith("_id") or nc == "id" or "id_" in nc or "_id_" in nc:
            continue
        candidates.append(c)

    if len(candidates) == 1:
        return smiles_col, seq_col, candidates[0]

    for c in ["y", "affinity", "label", "score"]:
        cc = {_normalize_colname(x): x for x in df.columns}.get(_normalize_colname(c))
        if cc is not None and pd.api.types.is_numeric_dtype(df[cc]):
            return smiles_col, seq_col, cc

    raise KeyError(
        "Cannot infer label column from pairs dataframe. "
        f"columns={list(df.columns)} numeric_candidates={candidates}"
    )


# ----------------------------
# GraphDTA-compatible featurizer
# ----------------------------

ATOM_LIST = [
    "C","N","O","S","F","Si","P","Cl","Br","Mg","Na","Ca","Fe","As","Al","I","B","V","K","Tl",
    "Yb","Sb","Sn","Ag","Pd","Co","Se","Ti","Zn","H","Li","Ge","Cu","Au","Ni","Cd","In","Mn",
    "Zr","Cr","Pt","Hg","Pb","Unknown"
]
DEGREE_LIST = list(range(0, 11))
NUMH_LIST = list(range(0, 11))
VALENCE_LIST = list(range(0, 11))

SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {ch: i + 1 for i, ch in enumerate(SEQ_VOCAB)}
MAX_SEQ_LEN = 1000

_RDLOG_DISABLED = False


def _one_hot(x, allowable: List) -> List[int]:
    if x not in allowable:
        x = allowable[-1]
    return [1 if x == s else 0 for s in allowable]


def atom_features(atom) -> np.ndarray:
    symbol = atom.GetSymbol()
    degree = atom.GetDegree()
    num_h = atom.GetTotalNumHs()
    try:
        valence = int(atom.GetValence(getExplicit=False))
    except Exception:
        valence = int(atom.GetImplicitValence())
    aromatic = 1 if atom.GetIsAromatic() else 0

    feats = []
    feats += _one_hot(symbol, ATOM_LIST)
    feats += _one_hot(degree, DEGREE_LIST)
    feats += _one_hot(num_h, NUMH_LIST)
    feats += _one_hot(valence, VALENCE_LIST)
    feats += [aromatic]
    arr = np.asarray(feats, dtype=np.float32)
    s = float(arr.sum())
    if s > 0:
        arr = arr / s
    return arr


def smile_to_graph(smile: str) -> Tuple[int, np.ndarray, np.ndarray]:
    global _RDLOG_DISABLED
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except Exception as e:
        raise RuntimeError("RDKit is required for SMILES featurization but is not available.") from e

    if not _RDLOG_DISABLED:
        try:
            RDLogger.DisableLog("rdApp.*")
        except Exception:
            pass
        _RDLOG_DISABLED = True

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")

    c_size = int(mol.GetNumAtoms())
    features = np.zeros((c_size, 78), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        features[i, :] = atom_features(atom)

    edges: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        edges.append((a, b))
        edges.append((b, a))

    edge_index = np.asarray(edges, dtype=np.int64).T if len(edges) > 0 else np.zeros((2, 0), dtype=np.int64)
    return c_size, features, edge_index


def seq_cat(seq: str) -> np.ndarray:
    x = np.zeros((MAX_SEQ_LEN,), dtype=np.int64)
    seq = seq.strip()
    n = min(len(seq), MAX_SEQ_LEN)
    for i in range(n):
        x[i] = SEQ_DICT.get(seq[i], 0)
    return x


class GraphPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_df: pd.DataFrame, indices: np.ndarray, smiles_col: str, seq_col: str, y_col: str):
        self.df = pairs_df.reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.smiles_col = smiles_col
        self.seq_col = seq_col
        self.y_col = y_col
        self._smiles_cache: Dict[str, Tuple[int, np.ndarray, np.ndarray]] = {}
        self._seq_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> GeoData:
        idx = int(self.indices[i])
        row = self.df.iloc[idx]
        smiles = str(row[self.smiles_col])
        seq = str(row[self.seq_col])
        y = float(row[self.y_col])

        if smiles in self._smiles_cache:
            c_size, features, edge_index = self._smiles_cache[smiles]
        else:
            c_size, features, edge_index = smile_to_graph(smiles)
            self._smiles_cache[smiles] = (c_size, features, edge_index)

        if seq in self._seq_cache:
            target = self._seq_cache[seq]
        else:
            target = seq_cat(seq)
            self._seq_cache[seq] = target

        x = torch.tensor(features, dtype=torch.float)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)
        y_t = torch.tensor([y], dtype=torch.float)

        # Important: make target 2D so PyG batches it into (B, 1000) instead of flattening to (B*1000,)
        target_t = torch.tensor(target, dtype=torch.long).unsqueeze(0)

        data = GeoData(x=x, edge_index=edge_index_t, y=y_t)
        data.target = target_t
        data.c_size = torch.tensor([c_size], dtype=torch.long)
        data.smiles = smiles
        data.sequence = seq
        return data


def build_model(model_name: str, graphdta_dir: Path) -> nn.Module:
    sys.path.insert(0, str(graphdta_dir))
    if model_name == "gin":
        from models.ginconv import GINConvNet  # type: ignore
        return GINConvNet()
    if model_name == "gat":
        from models.gat import GATNet  # type: ignore
        return GATNet()
    if model_name == "gat_gcn":
        from models.gat_gcn import GAT_GCN  # type: ignore
        return GAT_GCN()
    if model_name == "gcn":
        from models.gcn import GCNNet  # type: ignore
        return GCNNet()
    raise ValueError(f"Unknown model: {model_name}")


def make_run_dir(exp_name: str, out_subdir: Optional[str]) -> Path:
    root = repo_root()
    if out_subdir:
        p = Path(out_subdir)
        run_dir = p if p.is_absolute() else (root / p)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        run_dir = root / "runs" / date_str / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_yaml(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def to_preds_df(y: np.ndarray, yhat: np.ndarray, smiles: List[str], sequence: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": np.asarray(y, dtype=np.float64),
            "yhat": np.asarray(yhat, dtype=np.float64),
            "smiles": smiles,
            "sequence": sequence,
        }
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    model.eval()
    ys: List[float] = []
    yhats: List[float] = []
    smiles: List[str] = []
    seqs: List[str] = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch).view(-1).detach().cpu().numpy()
        y = batch.y.view(-1).detach().cpu().numpy()
        ys.extend(y.tolist())
        yhats.extend(out.tolist())

        bsz = int(batch.num_graphs)
        if hasattr(batch, "smiles") and hasattr(batch, "sequence"):
            if isinstance(batch.smiles, (list, tuple)) and len(batch.smiles) == bsz:
                smiles.extend([str(x) for x in batch.smiles])
            else:
                smiles.extend([""] * bsz)
            if isinstance(batch.sequence, (list, tuple)) and len(batch.sequence) == bsz:
                seqs.extend([str(x) for x in batch.sequence])
            else:
                seqs.extend([""] * bsz)
        else:
            smiles.extend([""] * bsz)
            seqs.extend([""] * bsz)

    return np.asarray(ys), np.asarray(yhats), smiles, seqs


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GraphDTA point predictor (post-hoc CP uses preds_*.csv.gz)")
    ap.add_argument("--dataset", choices=["davis", "kiba"], required=True)
    ap.add_argument("--split", choices=["random", "cold_drug", "cold_target", "cold_pair"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["gin", "gat", "gat_gcn", "gcn"], default="gat_gcn")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val_frac", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1, help="Kept for compatibility; not used by training.")
    ap.add_argument("--out_subdir", type=str, default=None, help="Optional output directory (relative to repo root).")
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    root = repo_root()
    graphdta_dir = find_graphdta_dir(root)
    graphdta_commit = get_git_commit(graphdta_dir)
    code_commit = get_git_commit(root)

    exp_name = f"graphdta_point_{args.model}_{args.dataset}_{args.split}_seed{args.seed}"
    run_dir = make_run_dir(exp_name, args.out_subdir)

    pairs_path = root / "data" / "processed" / args.dataset / "pairs.csv.gz"
    split_dir = root / "data" / "processed" / args.dataset / "splits" / args.split / f"seed_{args.seed}"

    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs file: {pairs_path}")
    if not split_dir.exists():
        parent = split_dir.parent
        hint = ""
        if parent.exists():
            siblings = sorted([p.name for p in parent.glob("seed_*") if p.is_dir()])[:20]
            hint = f"\nExisting seeds under {parent} (first 20): {siblings}"
        raise FileNotFoundError(f"Missing split directory: {split_dir}{hint}")

    with gzip.open(pairs_path, "rt") as f:
        pairs_df = pd.read_csv(f)

    smiles_col, seq_col, y_col = infer_pair_columns(pairs_df)
    splits = load_split_indices(split_dir)
    train_idx, cal_idx, test_idx = splits["train"], splits["cal"], splits["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = GraphPairDataset(pairs_df, train_idx, smiles_col, seq_col, y_col)
    cal_ds = GraphPairDataset(pairs_df, cal_idx, smiles_col, seq_col, y_col)
    test_ds = GraphPairDataset(pairs_df, test_idx, smiles_col, seq_col, y_col)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    model = build_model(args.model, graphdta_dir).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()

    best_cal_rmse = float("inf")
    best_state = None
    bad_epochs = 0

    config = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": int(args.seed),
        "model": args.model,
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "val_frac": float(args.val_frac),
        "num_workers": int(args.num_workers),
        "alpha": float(args.alpha),
        "device": str(device),
        "run_dir": str(run_dir),
        "pairs_path": str(pairs_path),
        "split_dir": str(split_dir),
        "pairs_columns": {"smiles": smiles_col, "sequence": seq_col, "y": y_col},
        "graphdta_commit": graphdta_commit,
        "code_commit": code_commit,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_yaml(run_dir / "config.yaml", config)

    print(f"[START] run_dir={run_dir}")
    print(f"[INFO] dataset={args.dataset} split={args.split} seed={args.seed} model={args.model} device={device}")
    print(f"[INFO] n_train={len(train_ds)} n_cal={len(cal_ds)} n_test={len(test_ds)}")
    print(f"[INFO] pairs columns: smiles={smiles_col} sequence={seq_col} y={y_col}")
    sys.stdout.flush()

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(batch).view(-1)
            y = batch.y.view(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        y_cal, yhat_cal, _, _ = predict(model, cal_loader, device)
        cal_rmse = rmse(y_cal, yhat_cal)

        if cal_rmse < best_cal_rmse:
            best_cal_rmse = cal_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"[EPOCH] {epoch:03d} cal_rmse={cal_rmse:.6f} best_cal_rmse={best_cal_rmse:.6f} bad_epochs={bad_epochs}")
        sys.stdout.flush()

        if bad_epochs >= int(args.patience):
            print(f"[STOP] early stopping at epoch={epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "model.pt")

    y_cal, yhat_cal, cal_smiles, cal_seq = predict(model, cal_loader, device)
    y_test, yhat_test, test_smiles, test_seq = predict(model, test_loader, device)

    preds_cal_df = to_preds_df(y_cal, yhat_cal, cal_smiles, cal_seq)
    preds_test_df = to_preds_df(y_test, yhat_test, test_smiles, test_seq)

    preds_cal_path = run_dir / "preds_cal.csv.gz"
    preds_test_path = run_dir / "preds_test.csv.gz"
    preds_cal_df.to_csv(preds_cal_path, index=False, compression="gzip")
    preds_test_df.to_csv(preds_test_path, index=False, compression="gzip")

    metrics = {
        "rmse_cal_point": rmse(y_cal, yhat_cal),
        "cindex_cal_point": cindex(y_cal, yhat_cal),
        "rmse_test_point": rmse(y_test, yhat_test),
        "cindex_test_point": cindex(y_test, yhat_test),
        "n_train": int(len(train_ds)),
        "n_cal": int(len(cal_ds)),
        "n_test": int(len(test_ds)),
        "graphdta_commit": graphdta_commit,
        "code_commit": code_commit,
    }
    save_json(run_dir / "metrics.json", metrics)

    print("[DONE] Saved:")
    print(f" - {run_dir / 'config.yaml'}")
    print(f" - {run_dir / 'model.pt'}")
    print(f" - {run_dir / 'metrics.json'}")
    print(f" - {preds_cal_path}")
    print(f" - {preds_test_path}")


if __name__ == "__main__":
    main()
