import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n < 2:
        return float(np.max(scores)) if n == 1 else 0.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_dataset_from_run(run_dir: Path) -> str:
    m = load_json(run_dir / "metrics.json")
    dataset = m.get("dataset")
    if dataset not in ("davis", "kiba"):
        raise ValueError("Cannot infer dataset from metrics.json")
    return dataset


def build_entity_maps(pairs_path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = pd.read_csv(pairs_path, compression="gzip")
    drug_map = df.groupby("drug_idx")["smiles"].first().to_dict()
    target_map = df.groupby("target_idx")["sequence"].first().to_dict()
    drug_map = {int(k): str(v) for k, v in drug_map.items()}
    target_map = {int(k): str(v) for k, v in target_map.items()}
    return drug_map, target_map


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)

    arr = np.zeros((n_bits,), dtype=np.int8)
    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)

    return arr.astype(np.float32)


def aa_composition(seq: str) -> np.ndarray:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    counts = np.zeros((len(alphabet),), dtype=np.float32)
    seq = seq.strip().upper()
    if len(seq) == 0:
        return counts
    idx = {c: i for i, c in enumerate(alphabet)}
    for ch in seq:
        i = idx.get(ch, None)
        if i is not None:
            counts[i] += 1.0
    counts /= float(len(seq))
    return counts


def fit_kmeans(features: np.ndarray, n_clusters: int, seed: int, pca_dim: int):
    X = features
    if pca_dim is not None and pca_dim > 0:
        max_dim = int(min(X.shape[0], X.shape[1]))
        dim = int(min(pca_dim, max_dim))
        if dim > 0 and X.shape[1] > dim:
            pca = PCA(n_components=dim, random_state=seed)
            X = pca.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X)
    return km, X


def transform_features(features: np.ndarray, seed: int, pca_dim: int, ref_features: np.ndarray) -> np.ndarray:
    X = features
    if pca_dim is not None and pca_dim > 0:
        max_dim = int(min(ref_features.shape[0], ref_features.shape[1]))
        dim = int(min(pca_dim, max_dim))
        if dim > 0 and ref_features.shape[1] > dim:
            pca = PCA(n_components=dim, random_state=seed)
            pca.fit(ref_features)
            X = pca.transform(features)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--group_by", type=str, choices=["drug", "target"], required=True)
    ap.add_argument("--n_clusters", type=int, default=50)
    ap.add_argument("--min_cal_per_cluster", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp_bits", type=int, default=2048)
    ap.add_argument("--fp_radius", type=int, default=2)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--out_subdir", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cal_path = run_dir / "preds_cal.csv.gz"
    test_path = run_dir / "preds_test.csv.gz"
    if not cal_path.exists() or not test_path.exists():
        raise FileNotFoundError("preds_cal.csv.gz or preds_test.csv.gz missing")

    dataset = get_dataset_from_run(run_dir)
    pairs_path = Path("data") / "processed" / dataset / "pairs.csv.gz"
    drug_map, target_map = build_entity_maps(pairs_path)

    cal = pd.read_csv(cal_path, compression="gzip")
    test = pd.read_csv(test_path, compression="gzip")

    for col in ["y", "y_pred", "drug_idx", "target_idx"]:
        if col not in cal.columns or col not in test.columns:
            raise ValueError(f"missing required column '{col}' in preds files")

    cal_scores = np.abs(cal["y"].to_numpy(dtype=np.float64) - cal["y_pred"].to_numpy(dtype=np.float64))
    global_qhat = conformal_qhat(cal_scores, args.alpha)

    if args.group_by == "drug":
        cal_entities = sorted(set(int(x) for x in cal["drug_idx"].tolist()))
        test_entities = sorted(set(int(x) for x in test["drug_idx"].tolist()))
        all_entities = sorted(set(cal_entities) | set(test_entities))

        feats = []
        for eid in tqdm(all_entities, desc="featurize_drugs"):
            smi = drug_map.get(eid, "")
            feats.append(morgan_fp(smi, n_bits=args.fp_bits, radius=args.fp_radius))
        feats = np.stack(feats, axis=0)
    else:
        cal_entities = sorted(set(int(x) for x in cal["target_idx"].tolist()))
        test_entities = sorted(set(int(x) for x in test["target_idx"].tolist()))
        all_entities = sorted(set(cal_entities) | set(test_entities))

        feats = []
        for eid in tqdm(all_entities, desc="featurize_targets"):
            seq = target_map.get(eid, "")
            feats.append(aa_composition(seq))
        feats = np.stack(feats, axis=0)

    ent_to_row = {eid: i for i, eid in enumerate(all_entities)}

    cal_idx = np.array([ent_to_row[int(e)] for e in (cal["drug_idx"] if args.group_by == "drug" else cal["target_idx"])], dtype=np.int64)
    test_idx = np.array([ent_to_row[int(e)] for e in (test["drug_idx"] if args.group_by == "drug" else test["target_idx"])], dtype=np.int64)

    ref_feats = feats[np.array([ent_to_row[e] for e in cal_entities], dtype=np.int64)]
    km, ref_X = fit_kmeans(ref_feats, n_clusters=args.n_clusters, seed=args.seed, pca_dim=args.pca_dim)

    X_all = transform_features(feats, seed=args.seed, pca_dim=args.pca_dim, ref_features=ref_feats)
    labels_all = km.predict(X_all)

    cal_labels = labels_all[cal_idx]
    test_labels = labels_all[test_idx]

    cal_scores = np.asarray(cal_scores, dtype=np.float64)

    qhat_by_cluster: Dict[int, float] = {}
    n_by_cluster: Dict[int, int] = {}

    for c in range(args.n_clusters):
        mask = (cal_labels == c)
        n = int(np.sum(mask))
        n_by_cluster[c] = n
        if n >= args.min_cal_per_cluster:
            qhat_by_cluster[c] = conformal_qhat(cal_scores[mask], args.alpha)
        else:
            qhat_by_cluster[c] = global_qhat

    qhat_test = np.array([qhat_by_cluster[int(c)] for c in test_labels], dtype=np.float64)
    qhat_test = np.maximum(qhat_test, global_qhat)


    y_test = test["y"].to_numpy(dtype=np.float64)
    p_test = test["y_pred"].to_numpy(dtype=np.float64)
    lo = p_test - qhat_test
    hi = p_test + qhat_test
    covered = (y_test >= lo) & (y_test <= hi)

    metrics = {
        "alpha": float(args.alpha),
        "group_by": args.group_by,
        "n_clusters": int(args.n_clusters),
        "min_cal_per_cluster": int(args.min_cal_per_cluster),
        "global_qhat": float(global_qhat),
        "coverage_test": float(np.mean(covered)),
        "avg_width_test": float(np.mean(hi - lo)),
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
        "clusters_with_fallback": int(sum(1 for c in range(args.n_clusters) if n_by_cluster[c] < args.min_cal_per_cluster)),
        "median_cal_per_cluster": float(np.median(list(n_by_cluster.values()))),
    }

    if args.out_subdir is None:
        out_subdir = f"cp_cluster_{args.group_by}_K{args.n_clusters}_alpha{args.alpha}"
    else:
        out_subdir = args.out_subdir

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "conformal_metrics.json", metrics)

    out = test.copy()
    out["cluster"] = test_labels.astype(int)
    out["qhat"] = qhat_test
    out["pi_lo"] = lo
    out["pi_hi"] = hi
    out["covered"] = covered.astype(int)
    out.to_csv(out_dir / "pred_intervals_test.csv.gz", index=False, compression="gzip")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
