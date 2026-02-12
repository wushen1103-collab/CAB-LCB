import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required for this script.") from e

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
except Exception as e:
    raise RuntimeError("scikit-learn is required for this script.") from e

try:
    from rdkit import Chem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
except Exception:
    Chem = None
    GetMorganGenerator = None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _decode_if_bytes(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == object:
        out = []
        for x in arr.tolist():
            if isinstance(x, (bytes, bytearray)):
                out.append(x.decode("utf-8"))
            else:
                out.append(x)
        return np.asarray(out, dtype=object)
    if arr.dtype.kind in ("S",):
        return np.asarray([x.decode("utf-8") for x in arr.tolist()], dtype=object)
    return arr


def _pick_col(df: "pd.DataFrame", options: List[str]) -> Optional[str]:
    cols = set(df.columns.tolist())
    for c in options:
        if c in cols:
            return c
    return None


def _find_first_existing(run_dir: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = run_dir / n
        if p.exists() and p.is_file():
            return p
    return None


def _load_from_npz_files(run_dir: Path, key_candidates: List[str]) -> Optional[np.ndarray]:
    npz_files = sorted(run_dir.glob("*.npz"))
    for npz_path in npz_files:
        try:
            z = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue
        for k in key_candidates:
            if k in z:
                arr = z[k]
                arr = _decode_if_bytes(np.asarray(arr))
                return arr
    return None


def load_artifact(run_dir: Path, kind: str) -> np.ndarray:
    kind = kind.lower().strip()

    candidates = {
        "y_cal": ["y_cal.npy", "cal_y.npy", "y_valid.npy", "valid_y.npy", "y_val.npy", "val_y.npy"],
        "y_test": ["y_test.npy", "test_y.npy"],
        "pred_cal": ["pred_cal.npy", "cal_pred.npy", "pred_valid.npy", "valid_pred.npy", "pred_val.npy", "val_pred.npy"],
        "pred_test": ["pred_test.npy", "test_pred.npy"],
        "drug_idx_cal": ["drug_idx_cal.npy", "cal_drug_idx.npy", "drug_cal.npy"],
        "drug_idx_test": ["drug_idx_test.npy", "test_drug_idx.npy", "drug_test.npy"],
        "target_idx_cal": ["target_idx_cal.npy", "cal_target_idx.npy", "target_cal.npy"],
        "target_idx_test": ["target_idx_test.npy", "test_target_idx.npy", "target_test.npy"],
        "smiles_cal": ["smiles_cal.npy", "cal_smiles.npy"],
        "smiles_test": ["smiles_test.npy", "test_smiles.npy"],
        "seq_cal": ["seq_cal.npy", "sequence_cal.npy", "cal_seq.npy", "cal_sequence.npy"],
        "seq_test": ["seq_test.npy", "sequence_test.npy", "test_seq.npy", "test_sequence.npy"],
        "pair_idx_cal": ["pair_idx_cal.npy", "cal_pair_idx.npy", "idx_cal.npy"],
        "pair_idx_test": ["pair_idx_test.npy", "test_pair_idx.npy", "idx_test.npy"],
    }

    npz_keys = {
        "y_cal": ["y_cal", "cal_y", "y_valid", "y_val", "valid_y", "val_y"],
        "y_test": ["y_test", "test_y"],
        "pred_cal": ["pred_cal", "cal_pred", "pred_valid", "pred_val", "valid_pred", "val_pred"],
        "pred_test": ["pred_test", "test_pred"],
        "drug_idx_cal": ["drug_idx_cal", "cal_drug_idx", "drug_cal"],
        "drug_idx_test": ["drug_idx_test", "test_drug_idx", "drug_test"],
        "target_idx_cal": ["target_idx_cal", "cal_target_idx", "target_cal"],
        "target_idx_test": ["target_idx_test", "test_target_idx", "target_test"],
        "smiles_cal": ["smiles_cal", "cal_smiles"],
        "smiles_test": ["smiles_test", "test_smiles"],
        "seq_cal": ["seq_cal", "sequence_cal", "cal_seq", "cal_sequence"],
        "seq_test": ["seq_test", "sequence_test", "test_seq", "test_sequence"],
        "pair_idx_cal": ["pair_idx_cal", "cal_pair_idx", "idx_cal"],
        "pair_idx_test": ["pair_idx_test", "test_pair_idx", "idx_test"],
    }

    if kind not in candidates:
        raise ValueError(f"Unknown artifact kind: {kind}")

    p = _find_first_existing(run_dir, candidates[kind])
    if p is not None:
        if p.suffix.lower() == ".npy":
            arr = np.load(p, allow_pickle=True)
            return _decode_if_bytes(np.asarray(arr))
        raise ValueError(f"Unsupported artifact extension: {p}")

    arr = _load_from_npz_files(run_dir, npz_keys[kind])
    if arr is not None:
        return arr

    raise FileNotFoundError(f"Missing artifact '{kind}' under run_dir={run_dir}")


def load_pairs_side_info(dataset: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    pairs_path = Path("data") / "processed" / dataset / "pairs.csv.gz"
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs file not found: {pairs_path}")

    df = pd.read_csv(
        pairs_path,
        compression="gzip",
        usecols=["drug_idx", "target_idx", "smiles", "sequence"],
    )
    df["smiles"] = df["smiles"].astype(str)
    df["sequence"] = df["sequence"].astype(str)

    ddf = df.drop_duplicates(subset=["drug_idx"])[["drug_idx", "smiles"]]
    tdf = df.drop_duplicates(subset=["target_idx"])[["target_idx", "sequence"]]

    drug_map = {int(k): str(v) for k, v in zip(ddf["drug_idx"].tolist(), ddf["smiles"].tolist())}
    target_map = {int(k): str(v) for k, v in zip(tdf["target_idx"].tolist(), tdf["sequence"].tolist())}
    return drug_map, target_map


def load_from_preds_csv(run_dir: Path) -> Dict[str, np.ndarray]:
    cal_path = run_dir / "preds_cal.csv.gz"
    test_path = run_dir / "preds_test.csv.gz"
    if not cal_path.exists() or not test_path.exists():
        raise FileNotFoundError("preds_cal.csv.gz / preds_test.csv.gz not found under run_dir")

    dfc = pd.read_csv(cal_path, compression="gzip")
    dft = pd.read_csv(test_path, compression="gzip")

    y_col = _pick_col(dfc, ["y", "y_true", "label"])
    p_col = _pick_col(dfc, ["pred", "y_pred", "prediction"])
    if y_col is None or p_col is None:
        raise RuntimeError("Cannot locate y/pred columns in preds_cal.csv.gz")

    y_col_t = _pick_col(dft, ["y", "y_true", "label"])
    p_col_t = _pick_col(dft, ["pred", "y_pred", "prediction"])
    if y_col_t is None or p_col_t is None:
        raise RuntimeError("Cannot locate y/pred columns in preds_test.csv.gz")

    drug_col = _pick_col(dfc, ["drug_idx", "drug"])
    target_col = _pick_col(dfc, ["target_idx", "target"])
    if drug_col is None or target_col is None:
        raise RuntimeError("Cannot locate drug_idx/target_idx columns in preds_cal.csv.gz")

    drug_col_t = _pick_col(dft, ["drug_idx", "drug"])
    target_col_t = _pick_col(dft, ["target_idx", "target"])
    if drug_col_t is None or target_col_t is None:
        raise RuntimeError("Cannot locate drug_idx/target_idx columns in preds_test.csv.gz")

    smiles_col = _pick_col(dfc, ["smiles", "drug_smiles"])
    seq_col = _pick_col(dfc, ["sequence", "seq", "target_sequence"])
    smiles_col_t = _pick_col(dft, ["smiles", "drug_smiles"])
    seq_col_t = _pick_col(dft, ["sequence", "seq", "target_sequence"])

    pair_col = _pick_col(dfc, ["pair_idx", "row_idx", "idx", "example_idx"])
    pair_col_t = _pick_col(dft, ["pair_idx", "row_idx", "idx", "example_idx"])

    out = {
        "y_cal": dfc[y_col].to_numpy(dtype=np.float64),
        "pred_cal": dfc[p_col].to_numpy(dtype=np.float64),
        "drug_idx_cal": dfc[drug_col].to_numpy(dtype=np.int64),
        "target_idx_cal": dfc[target_col].to_numpy(dtype=np.int64),

        "y_test": dft[y_col_t].to_numpy(dtype=np.float64),
        "pred_test": dft[p_col_t].to_numpy(dtype=np.float64),
        "drug_idx_test": dft[drug_col_t].to_numpy(dtype=np.int64),
        "target_idx_test": dft[target_col_t].to_numpy(dtype=np.int64),
    }

    if smiles_col is not None:
        out["smiles_cal"] = dfc[smiles_col].astype(str).to_numpy(dtype=object)
    if seq_col is not None:
        out["seq_cal"] = dfc[seq_col].astype(str).to_numpy(dtype=object)
    if smiles_col_t is not None:
        out["smiles_test"] = dft[smiles_col_t].astype(str).to_numpy(dtype=object)
    if seq_col_t is not None:
        out["seq_test"] = dft[seq_col_t].astype(str).to_numpy(dtype=object)

    if pair_col is not None:
        out["pair_idx_cal"] = dfc[pair_col].to_numpy(dtype=np.int64)
    if pair_col_t is not None:
        out["pair_idx_test"] = dft[pair_col_t].to_numpy(dtype=np.int64)

    return out


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n <= 0:
        raise ValueError("Empty scores for conformal quantile.")
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def format_float_tag(x: float, decimals: int = 2) -> str:
    s = f"{float(x):.{decimals}f}"
    return s.replace(".", "p")


def build_entity_text(entity_ids: np.ndarray, texts: np.ndarray) -> Dict[int, str]:
    entity_ids = np.asarray(entity_ids, dtype=np.int64)
    texts = _decode_if_bytes(np.asarray(texts))
    out: Dict[int, str] = {}
    for eid, t in zip(entity_ids.tolist(), texts.tolist()):
        if eid not in out:
            out[eid] = str(t)
    return out


def aacomp_features(seqs: List[str]) -> np.ndarray:
    aa = "ACDEFGHIKLMNPQRSTVWY"
    idx = {c: i for i, c in enumerate(aa)}
    X = np.zeros((len(seqs), len(aa)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = (s or "").strip().upper()
        if len(s) == 0:
            continue
        cnt = np.zeros(len(aa), dtype=np.float32)
        tot = 0.0
        for ch in s:
            if ch in idx:
                cnt[idx[ch]] += 1.0
                tot += 1.0
        if tot > 0:
            cnt /= tot
        X[i] = cnt
    return X


def morgan_features(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    if Chem is None or GetMorganGenerator is None:
        raise RuntimeError("RDKit is required for Morgan fingerprints.")
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    for i, smi in enumerate(smiles_list):
        smi = (smi or "").strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        ConvertToNumpyArray(fp, arr)
        X[i] = arr.astype(np.float32)
    return X


def fit_transform_features(
    cal_texts: List[str],
    test_texts: List[str],
    repr_name: str,
    tfidf_ngram_min: int,
    tfidf_ngram_max: int,
    tfidf_max_features: int,
    pca_dim: int,
    seed: int,
) -> Tuple[object, object, int]:
    repr_name = repr_name.lower().strip()

    if repr_name == "tfidf":
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(int(tfidf_ngram_min), int(tfidf_ngram_max)),
            max_features=int(tfidf_max_features),
        )
        X_cal = vec.fit_transform(cal_texts)
        X_test = vec.transform(test_texts)
        return X_cal, X_test, 0

    if repr_name == "aacomp":
        X_cal = aacomp_features(cal_texts)
        X_test = aacomp_features(test_texts)
        if pca_dim and pca_dim > 0 and pca_dim < X_cal.shape[1]:
            pca = PCA(n_components=int(pca_dim), random_state=int(seed))
            X_cal = pca.fit_transform(X_cal).astype(np.float32)
            X_test = pca.transform(X_test).astype(np.float32)
            return X_cal, X_test, int(pca_dim)
        return X_cal, X_test, 0

    if repr_name == "morgan":
        X_cal = morgan_features(cal_texts)
        X_test = morgan_features(test_texts)
        if pca_dim and pca_dim > 0 and pca_dim < X_cal.shape[1]:
            pca = PCA(n_components=int(pca_dim), random_state=int(seed))
            X_cal = pca.fit_transform(X_cal).astype(np.float32)
            X_test = pca.transform(X_test).astype(np.float32)
            return X_cal, X_test, int(pca_dim)
        return X_cal, X_test, 0

    raise ValueError(f"Unknown representation: {repr_name}")


def select_by_full_indices(
    arr: np.ndarray,
    arr_full_idx: Optional[np.ndarray],
    desired_full_idx: np.ndarray,
    fallback_ordered_full_idx: Optional[np.ndarray],
) -> np.ndarray:
    desired_full_idx = np.unique(np.asarray(desired_full_idx, dtype=np.int64))

    if arr_full_idx is None:
        if fallback_ordered_full_idx is None:
            raise ValueError("No full index mapping available for selection.")
        arr_full_idx = np.asarray(fallback_ordered_full_idx, dtype=np.int64)

    arr_full_idx = np.asarray(arr_full_idx, dtype=np.int64)
    pos_map = {int(i): int(p) for p, i in enumerate(arr_full_idx.tolist())}

    positions = []
    for i in desired_full_idx.tolist():
        if int(i) not in pos_map:
            raise ValueError(f"desired index {int(i)} is missing in available indices")
        positions.append(pos_map[int(i)])

    positions = np.asarray(positions, dtype=np.int64)
    return arr[positions]


@dataclass
class ConformalMetrics:
    alpha: float
    group_by: str
    k_neighbors: int
    min_cal_samples: int
    drug_repr: str
    target_repr: str
    pca_dim_used: int
    knn_metric: str
    dist_norm: str
    distance_inflate_gamma: float
    global_qhat: float
    coverage_eval: float
    avg_width_eval: float
    n_cal_used: int
    n_eval_used: int
    n_eval_entities: int
    n_eval_entities_used_global: int
    qhat_eval_entity_min: float
    qhat_eval_entity_median: float
    qhat_eval_entity_max: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--group_by", type=str, choices=["drug", "target"], required=True)

    ap.add_argument("--k_neighbors", type=int, default=60)
    ap.add_argument("--min_cal_samples", type=int, default=200)

    ap.add_argument("--drug_repr", type=str, choices=["morgan"], default="morgan")
    ap.add_argument("--target_repr", type=str, choices=["aacomp", "tfidf"], default="aacomp")

    ap.add_argument("--tfidf_ngram_min", type=int, default=3)
    ap.add_argument("--tfidf_ngram_max", type=int, default=5)
    ap.add_argument("--tfidf_max_features", type=int, default=8000)

    ap.add_argument("--knn_metric", type=str, choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--dist_norm", type=str, choices=["none", "median"], default="median")
    ap.add_argument("--distance_inflate_gamma", type=float, default=0.0)

    ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--out_subdir", type=str, default="")

    ap.add_argument(
        "--label_transform",
        type=str,
        default="none",
        choices=["none", "pkd"],
        help="Optional label/prediction transform applied before conformal. "
             "pkd is applied only for Davis (Kd[nM] -> pKd).",
    )

    ap.add_argument("--cal_idx_npy", type=str, default="")
    ap.add_argument("--eval_idx_npy", type=str, default="")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        raise FileNotFoundError(f"metrics.json not found under run_dir: {run_dir}")
    m = load_json(mpath)

    dataset = str(m.get("dataset", ""))
    split = str(m.get("split", ""))
    seed = int(m.get("seed", 0))

    split_dir = Path("data") / "processed" / dataset / "splits" / split / f"seed_{seed}"
    idx_cal_path = split_dir / "idx_cal.npy"
    idx_test_path = split_dir / "idx_test.npy"
    if not idx_cal_path.exists() or not idx_test_path.exists():
        raise FileNotFoundError(f"Expected split index files not found under: {split_dir}")

    idx_cal_full = np.load(idx_cal_path).astype(np.int64)
    idx_test_full = np.load(idx_test_path).astype(np.int64)

    drug_map, target_map = load_pairs_side_info(dataset)

    try:
        y_cal_full = np.asarray(load_artifact(run_dir, "y_cal"), dtype=np.float64)
        pred_cal_full = np.asarray(load_artifact(run_dir, "pred_cal"), dtype=np.float64)
        y_eval_full = np.asarray(load_artifact(run_dir, "y_test"), dtype=np.float64)
        pred_eval_full = np.asarray(load_artifact(run_dir, "pred_test"), dtype=np.float64)

        drug_idx_cal_full = np.asarray(load_artifact(run_dir, "drug_idx_cal"), dtype=np.int64)
        target_idx_cal_full = np.asarray(load_artifact(run_dir, "target_idx_cal"), dtype=np.int64)
        drug_idx_eval_full = np.asarray(load_artifact(run_dir, "drug_idx_test"), dtype=np.int64)
        target_idx_eval_full = np.asarray(load_artifact(run_dir, "target_idx_test"), dtype=np.int64)

        try:
            smiles_cal_full = _decode_if_bytes(load_artifact(run_dir, "smiles_cal"))
        except Exception:
            smiles_cal_full = np.asarray([drug_map.get(int(i), "") for i in drug_idx_cal_full.tolist()], dtype=object)

        try:
            seq_cal_full = _decode_if_bytes(load_artifact(run_dir, "seq_cal"))
        except Exception:
            seq_cal_full = np.asarray([target_map.get(int(i), "") for i in target_idx_cal_full.tolist()], dtype=object)

        try:
            smiles_eval_full = _decode_if_bytes(load_artifact(run_dir, "smiles_test"))
        except Exception:
            smiles_eval_full = np.asarray([drug_map.get(int(i), "") for i in drug_idx_eval_full.tolist()], dtype=object)

        try:
            seq_eval_full = _decode_if_bytes(load_artifact(run_dir, "seq_test"))
        except Exception:
            seq_eval_full = np.asarray([target_map.get(int(i), "") for i in target_idx_eval_full.tolist()], dtype=object)

        pair_idx_cal_full = None
        pair_idx_eval_full = None
        try:
            pair_idx_cal_full = np.asarray(load_artifact(run_dir, "pair_idx_cal"), dtype=np.int64)
        except Exception:
            pair_idx_cal_full = None
        try:
            pair_idx_eval_full = np.asarray(load_artifact(run_dir, "pair_idx_test"), dtype=np.int64)
        except Exception:
            pair_idx_eval_full = None

    except FileNotFoundError:
        blob = load_from_preds_csv(run_dir)

        y_cal_full = blob["y_cal"]
        pred_cal_full = blob["pred_cal"]
        drug_idx_cal_full = blob["drug_idx_cal"]
        target_idx_cal_full = blob["target_idx_cal"]

        y_eval_full = blob["y_test"]
        pred_eval_full = blob["pred_test"]
        drug_idx_eval_full = blob["drug_idx_test"]
        target_idx_eval_full = blob["target_idx_test"]

        smiles_cal_full = blob.get("smiles_cal", None)
        seq_cal_full = blob.get("seq_cal", None)
        smiles_eval_full = blob.get("smiles_test", None)
        seq_eval_full = blob.get("seq_test", None)

        if smiles_cal_full is None:
            smiles_cal_full = np.asarray([drug_map.get(int(i), "") for i in drug_idx_cal_full.tolist()], dtype=object)
        if seq_cal_full is None:
            seq_cal_full = np.asarray([target_map.get(int(i), "") for i in target_idx_cal_full.tolist()], dtype=object)
        if smiles_eval_full is None:
            smiles_eval_full = np.asarray([drug_map.get(int(i), "") for i in drug_idx_eval_full.tolist()], dtype=object)
        if seq_eval_full is None:
            seq_eval_full = np.asarray([target_map.get(int(i), "") for i in target_idx_eval_full.tolist()], dtype=object)

        pair_idx_cal_full = blob.get("pair_idx_cal", None)
        pair_idx_eval_full = blob.get("pair_idx_test", None)

    # Optional label/prediction transform (applied consistently to cal and eval arrays).
    label_space = "raw"
    if args.label_transform == "pkd":
        ds_lower = str(dataset).lower() if dataset is not None else ""
        if ds_lower == "davis":
            label_space = "pkd"

            def _to_pkd(arr):
                # Convert Kd in nM to pKd: pKd = 9 - log10(Kd[nM]).
                import numpy as np
                arr = np.asarray(arr, dtype=np.float64)
                arr = np.clip(arr, 1e-12, None)
                return 9.0 - np.log10(arr)

            y_cal_full = _to_pkd(y_cal_full)
            pred_cal_full = _to_pkd(pred_cal_full)
            y_eval_full = _to_pkd(y_eval_full)
            pred_eval_full = _to_pkd(pred_eval_full)
        else:
            # For non-Davis datasets, pkd transform is a no-op by design.
            print("[WARN] --label_transform=pkd requested, but dataset is not Davis; leaving labels unchanged.")

    y_cal = y_cal_full
    pred_cal = pred_cal_full
    drug_idx_cal = drug_idx_cal_full
    target_idx_cal = target_idx_cal_full
    smiles_cal = smiles_cal_full
    seq_cal = seq_cal_full

    y_eval = y_eval_full
    pred_eval = pred_eval_full
    drug_idx_eval = drug_idx_eval_full
    target_idx_eval = target_idx_eval_full
    smiles_eval = smiles_eval_full
    seq_eval = seq_eval_full

    if args.cal_idx_npy:
        desired = np.load(Path(args.cal_idx_npy)).astype(np.int64)
        y_cal = select_by_full_indices(y_cal_full, pair_idx_cal_full, desired, idx_cal_full)
        pred_cal = select_by_full_indices(pred_cal_full, pair_idx_cal_full, desired, idx_cal_full)
        drug_idx_cal = select_by_full_indices(drug_idx_cal_full, pair_idx_cal_full, desired, idx_cal_full)
        target_idx_cal = select_by_full_indices(target_idx_cal_full, pair_idx_cal_full, desired, idx_cal_full)
        smiles_cal = select_by_full_indices(smiles_cal_full, pair_idx_cal_full, desired, idx_cal_full)
        seq_cal = select_by_full_indices(seq_cal_full, pair_idx_cal_full, desired, idx_cal_full)

    if args.eval_idx_npy:
        desired = np.load(Path(args.eval_idx_npy)).astype(np.int64)
        try:
            y_eval = select_by_full_indices(y_eval_full, pair_idx_eval_full, desired, idx_test_full)
            pred_eval = select_by_full_indices(pred_eval_full, pair_idx_eval_full, desired, idx_test_full)
            drug_idx_eval = select_by_full_indices(drug_idx_eval_full, pair_idx_eval_full, desired, idx_test_full)
            target_idx_eval = select_by_full_indices(target_idx_eval_full, pair_idx_eval_full, desired, idx_test_full)
            smiles_eval = select_by_full_indices(smiles_eval_full, pair_idx_eval_full, desired, idx_test_full)
            seq_eval = select_by_full_indices(seq_eval_full, pair_idx_eval_full, desired, idx_test_full)
        except ValueError:
            y_eval = select_by_full_indices(y_cal_full, pair_idx_cal_full, desired, idx_cal_full)
            pred_eval = select_by_full_indices(pred_cal_full, pair_idx_cal_full, desired, idx_cal_full)
            drug_idx_eval = select_by_full_indices(drug_idx_cal_full, pair_idx_cal_full, desired, idx_cal_full)
            target_idx_eval = select_by_full_indices(target_idx_cal_full, pair_idx_cal_full, desired, idx_cal_full)
            smiles_eval = select_by_full_indices(smiles_cal_full, pair_idx_cal_full, desired, idx_cal_full)
            seq_eval = select_by_full_indices(seq_cal_full, pair_idx_cal_full, desired, idx_cal_full)

    scores_cal = np.abs(y_cal - pred_cal)
    global_qhat = conformal_qhat(scores_cal, float(args.alpha))

    if args.group_by == "drug":
        cal_entity_ids = np.unique(drug_idx_cal).astype(np.int64)
        eval_entity_ids = np.unique(drug_idx_eval).astype(np.int64)

        cal_map = build_entity_text(drug_idx_cal, smiles_cal)
        eval_map = build_entity_text(drug_idx_eval, smiles_eval)

        cal_texts = [cal_map[int(e)] for e in cal_entity_ids.tolist()]
        eval_texts = [eval_map[int(e)] for e in eval_entity_ids.tolist()]

        X_cal, X_eval, pca_dim_used = fit_transform_features(
            cal_texts=cal_texts,
            test_texts=eval_texts,
            repr_name=args.drug_repr,
            tfidf_ngram_min=args.tfidf_ngram_min,
            tfidf_ngram_max=args.tfidf_ngram_max,
            tfidf_max_features=args.tfidf_max_features,
            pca_dim=args.pca_dim,
            seed=seed,
        )

        entity_cal_for_samples = drug_idx_cal
        entity_eval_for_samples = drug_idx_eval

    else:
        cal_entity_ids = np.unique(target_idx_cal).astype(np.int64)
        eval_entity_ids = np.unique(target_idx_eval).astype(np.int64)

        cal_map = build_entity_text(target_idx_cal, seq_cal)
        eval_map = build_entity_text(target_idx_eval, seq_eval)

        cal_texts = [cal_map[int(e)] for e in cal_entity_ids.tolist()]
        eval_texts = [eval_map[int(e)] for e in eval_entity_ids.tolist()]

        X_cal, X_eval, pca_dim_used = fit_transform_features(
            cal_texts=cal_texts,
            test_texts=eval_texts,
            repr_name=args.target_repr,
            tfidf_ngram_min=args.tfidf_ngram_min,
            tfidf_ngram_max=args.tfidf_ngram_max,
            tfidf_max_features=args.tfidf_max_features,
            pca_dim=args.pca_dim,
            seed=seed,
        )

        entity_cal_for_samples = target_idx_cal
        entity_eval_for_samples = target_idx_eval

    cal_entity_to_scores: Dict[int, np.ndarray] = {}
    for eid in cal_entity_ids.tolist():
        mask = (entity_cal_for_samples == int(eid))
        cal_entity_to_scores[int(eid)] = scores_cal[mask]

    nn_algo = "brute" if args.knn_metric == "cosine" else "auto"
    nbrs = NearestNeighbors(
        n_neighbors=min(int(args.k_neighbors), len(cal_entity_ids)),
        metric=str(args.knn_metric),
        algorithm=nn_algo,
    )
    nbrs.fit(X_cal)

    dists, neigh_idx = nbrs.kneighbors(X_eval, return_distance=True)
    mean_dist = dists.mean(axis=1).astype(np.float64)

    mean_dist_norm = mean_dist.copy()
    if str(args.dist_norm) == "median":
        med = float(np.median(mean_dist_norm)) if len(mean_dist_norm) > 0 else 1.0
        med = med if med > 0 else 1.0
        mean_dist_norm = mean_dist_norm / med

    base_qhat = np.zeros((len(eval_entity_ids),), dtype=np.float64)
    used_global = 0

    for j in range(len(eval_entity_ids)):
        neigh_entities = cal_entity_ids[neigh_idx[j]].tolist()
        pooled = []
        for nid in neigh_entities:
            pooled.append(cal_entity_to_scores[int(nid)])
        pooled_scores = np.concatenate(pooled, axis=0) if len(pooled) > 0 else np.asarray([], dtype=np.float64)

        if pooled_scores.shape[0] < int(args.min_cal_samples):
            base_qhat[j] = float(global_qhat)
            used_global += 1
        else:
            base_qhat[j] = float(conformal_qhat(pooled_scores, float(args.alpha)))

    qhat_per_entity = base_qhat.copy()
    if float(args.distance_inflate_gamma) > 0.0:
        qhat_per_entity = qhat_per_entity * (1.0 + float(args.distance_inflate_gamma) * mean_dist_norm)

    entity_to_qhat: Dict[int, float] = {}
    for eid, q in zip(eval_entity_ids.tolist(), qhat_per_entity.tolist()):
        entity_to_qhat[int(eid)] = float(q)

    qhat_samples = np.asarray([entity_to_qhat[int(e)] for e in entity_eval_for_samples.tolist()], dtype=np.float64)

    err = np.abs(y_eval - pred_eval)
    coverage = float(np.mean(err <= qhat_samples))
    avg_width = float(np.mean(2.0 * qhat_samples))

    qmin = float(np.min(qhat_per_entity)) if len(qhat_per_entity) > 0 else float(global_qhat)
    qmed = float(np.median(qhat_per_entity)) if len(qhat_per_entity) > 0 else float(global_qhat)
    qmax = float(np.max(qhat_per_entity)) if len(qhat_per_entity) > 0 else float(global_qhat)

    metrics = ConformalMetrics(
        alpha=float(args.alpha),
        group_by=str(args.group_by),
        k_neighbors=int(args.k_neighbors),
        min_cal_samples=int(args.min_cal_samples),
        drug_repr=str(args.drug_repr),
        target_repr=str(args.target_repr),
        pca_dim_used=int(pca_dim_used),
        knn_metric=str(args.knn_metric),
        dist_norm=str(args.dist_norm),
        distance_inflate_gamma=float(args.distance_inflate_gamma),
        global_qhat=float(global_qhat),
        coverage_eval=float(coverage),
        avg_width_eval=float(avg_width),
        n_cal_used=int(len(y_cal)),
        n_eval_used=int(len(y_eval)),
        n_eval_entities=int(len(eval_entity_ids)),
        n_eval_entities_used_global=int(used_global),
        qhat_eval_entity_min=qmin,
        qhat_eval_entity_median=qmed,
        qhat_eval_entity_max=qmax,
    )

    if args.out_subdir:
        out_subdir = str(args.out_subdir)
    else:
        gamma_tag = format_float_tag(float(args.distance_inflate_gamma), decimals=2)
        out_subdir = (
            f"cp_local_{args.group_by}_{args.target_repr}"
            f"_k{int(args.k_neighbors)}_m{int(args.min_cal_samples)}"
            f"_gamma{gamma_tag}_alpha{float(args.alpha)}"
        )

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_metrics = asdict(metrics)
    out_metrics["label_space"] = label_space
    out_metrics["label_transform"] = str(args.label_transform)
    
    # ------------------------------------------------------------------
    # Write per-sample prediction intervals (needed by downstream eval).
    # ------------------------------------------------------------------
    try:
        df_pi = pd.DataFrame(
            {
                "drug_idx": drug_idx_eval.astype(np.int64),
                "target_idx": target_idx_eval.astype(np.int64),
                "y_true": y_eval.astype(np.float64),
                "pred": pred_eval.astype(np.float64),
                "qhat": qhat_samples.astype(np.float64),
                "pi_lo": (pred_eval - qhat_samples).astype(np.float64),
                "pi_hi": (pred_eval + qhat_samples).astype(np.float64),
            }
        )
        out_pi = out_dir / "pred_intervals_test.csv.gz"
        df_pi.to_csv(out_pi, index=False, compression="gzip")
        out_metrics["pred_intervals_test_path"] = str(out_pi)
        out_metrics["n_intervals_written"] = int(len(df_pi))
    except Exception as e:
        print(f"[WARN] failed to write pred_intervals_test.csv.gz: {e}")

    dump_json(out_dir / "conformal_metrics.json", out_metrics)

    cfg = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "args": vars(args),
    }
    dump_json(out_dir / "config.json", cfg)

    print(json.dumps(out_metrics, indent=2))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
