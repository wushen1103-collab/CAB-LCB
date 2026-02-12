import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


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


def clamp_pca_dim(pca_dim: int, n_samples: int, n_features: int) -> int:
    if pca_dim is None or pca_dim <= 0:
        return 0
    return int(min(pca_dim, n_samples, n_features))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--group_by", type=str, choices=["drug", "target"], required=True)
    ap.add_argument("--k_neighbors", type=int, default=20)
    ap.add_argument("--min_cal_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp_bits", type=int, default=2048)
    ap.add_argument("--fp_radius", type=int, default=2)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--distance_inflate_gamma", type=float, default=0.0)
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
        cal_eids = cal["drug_idx"].to_numpy(dtype=np.int64)
        test_eids = test["drug_idx"].to_numpy(dtype=np.int64)

        feats_cal = []
        for eid in tqdm(cal_entities, desc="featurize_cal_drugs"):
            feats_cal.append(morgan_fp(drug_map.get(eid, ""), n_bits=args.fp_bits, radius=args.fp_radius))
        feats_cal = np.stack(feats_cal, axis=0)

        feats_test = []
        for eid in tqdm(test_entities, desc="featurize_test_drugs"):
            feats_test.append(morgan_fp(drug_map.get(eid, ""), n_bits=args.fp_bits, radius=args.fp_radius))
        feats_test = np.stack(feats_test, axis=0)
    else:
        cal_entities = sorted(set(int(x) for x in cal["target_idx"].tolist()))
        test_entities = sorted(set(int(x) for x in test["target_idx"].tolist()))
        cal_eids = cal["target_idx"].to_numpy(dtype=np.int64)
        test_eids = test["target_idx"].to_numpy(dtype=np.int64)

        feats_cal = []
        for eid in tqdm(cal_entities, desc="featurize_cal_targets"):
            feats_cal.append(aa_composition(target_map.get(eid, "")))
        feats_cal = np.stack(feats_cal, axis=0)

        feats_test = []
        for eid in tqdm(test_entities, desc="featurize_test_targets"):
            feats_test.append(aa_composition(target_map.get(eid, "")))
        feats_test = np.stack(feats_test, axis=0)

    k = int(min(args.k_neighbors, len(cal_entities)))
    if k < 1:
        raise ValueError("k_neighbors must be >= 1 and cal_entities must be non-empty")

    pca_dim = clamp_pca_dim(args.pca_dim, n_samples=feats_cal.shape[0], n_features=feats_cal.shape[1])
    if pca_dim > 0 and feats_cal.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=args.seed)
        feats_cal_z = pca.fit_transform(feats_cal)
        feats_test_z = pca.transform(feats_test)
    else:
        feats_cal_z = feats_cal
        feats_test_z = feats_test

    # Map entity -> list of calibration scores
    scores_by_entity: Dict[int, List[float]] = {eid: [] for eid in cal_entities}
    for eid, sc in zip(cal_eids.tolist(), cal_scores.tolist()):
        scores_by_entity[int(eid)].append(float(sc))

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(feats_cal_z)

    dists, idxs = nbrs.kneighbors(feats_test_z, return_distance=True)

    # Compute entity-specific qhat for each test entity
    qhat_by_test_entity: Dict[int, float] = {}
    used_global = 0

    for j, test_eid in enumerate(test_entities):
        neigh_eids = [cal_entities[int(i)] for i in idxs[j].tolist()]
        local_scores = []
        for ne in neigh_eids:
            local_scores.extend(scores_by_entity.get(int(ne), []))

        if len(local_scores) < args.min_cal_samples:
            qhat = global_qhat
            used_global += 1
        else:
            qhat = conformal_qhat(np.array(local_scores, dtype=np.float64), args.alpha)

        # Optional distance inflation (off by default)
        if args.distance_inflate_gamma > 0:
            d = float(np.mean(dists[j]))
            qhat = qhat * (1.0 + args.distance_inflate_gamma * d)

        # Safe clamp: never smaller than global
        qhat = float(max(qhat, global_qhat))
        qhat_by_test_entity[int(test_eid)] = qhat

    # Assign per-sample qhat on test pairs
    if args.group_by == "drug":
        qhat_test = np.array([qhat_by_test_entity[int(e)] for e in test["drug_idx"].tolist()], dtype=np.float64)
    else:
        qhat_test = np.array([qhat_by_test_entity[int(e)] for e in test["target_idx"].tolist()], dtype=np.float64)

    y_test = test["y"].to_numpy(dtype=np.float64)
    p_test = test["y_pred"].to_numpy(dtype=np.float64)
    lo = p_test - qhat_test
    hi = p_test + qhat_test
    covered = (y_test >= lo) & (y_test <= hi)

    qhat_vals = np.array(list(qhat_by_test_entity.values()), dtype=np.float64)

    metrics = {
        "alpha": float(args.alpha),
        "group_by": args.group_by,
        "k_neighbors": int(k),
        "min_cal_samples": int(args.min_cal_samples),
        "pca_dim_used": int(pca_dim),
        "global_qhat": float(global_qhat),
        "coverage_test": float(np.mean(covered)),
        "avg_width_test": float(np.mean(hi - lo)),
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
        "n_test_entities": int(len(test_entities)),
        "n_test_entities_used_global": int(used_global),
        "qhat_test_entity_min": float(np.min(qhat_vals)) if len(qhat_vals) else float(global_qhat),
        "qhat_test_entity_median": float(np.median(qhat_vals)) if len(qhat_vals) else float(global_qhat),
        "qhat_test_entity_max": float(np.max(qhat_vals)) if len(qhat_vals) else float(global_qhat),
        "distance_inflate_gamma": float(args.distance_inflate_gamma),
    }

    if args.out_subdir is None:
        out_subdir = f"cp_knn_{args.group_by}_k{k}_alpha{args.alpha}"
    else:
        out_subdir = args.out_subdir

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "conformal_metrics.json", metrics)

    out = test.copy()
    out["qhat"] = qhat_test
    out["pi_lo"] = lo
    out["pi_hi"] = hi
    out["covered"] = covered.astype(int)
    out.to_csv(out_dir / "pred_intervals_test.csv.gz", index=False, compression="gzip")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
