from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from tqdm import tqdm

from src.dti_cp.data.dataset import PairDataset
from src.dti_cp.eval.metrics import rmse, concordance_index
from src.dti_cp.utils.io import make_run_dir, save_yaml


def load_split_indices(data_root: str, dataset: str, split: str, seed: int) -> Dict[str, np.ndarray]:
    base = Path(data_root) / "processed" / dataset / "splits" / split / f"seed_{seed}"
    idx_train = np.load(base / "idx_train.npy")
    idx_cal = np.load(base / "idx_cal.npy")
    idx_test = np.load(base / "idx_test.npy")
    return {"train": idx_train, "cal": idx_cal, "test": idx_test}


def _make_vectorizers(n_features: int, ngram_smiles: int, ngram_seq: int) -> Tuple[HashingVectorizer, HashingVectorizer]:
    vec_smiles = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        analyzer="char",
        ngram_range=(1, ngram_smiles),
    )
    vec_seq = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        analyzer="char",
        ngram_range=(1, ngram_seq),
    )
    return vec_smiles, vec_seq


def _transform_chunk(
    smiles_list: List[str],
    seq_list: List[str],
    n_features: int,
    ngram_smiles: int,
    ngram_seq: int,
) -> sparse.csr_matrix:
    vec_smiles, vec_seq = _make_vectorizers(n_features, ngram_smiles, ngram_seq)
    X_smi = vec_smiles.transform(smiles_list)
    X_seq = vec_seq.transform(seq_list)
    return sparse.hstack([X_smi, X_seq], format="csr")


def make_features_parallel(
    smiles: pd.Series,
    seq: pd.Series,
    n_features: int,
    ngram_smiles: int,
    ngram_seq: int,
    n_jobs: int,
) -> sparse.csr_matrix:
    smiles_list = smiles.astype(str).tolist()
    seq_list = seq.astype(str).tolist()
    n = len(smiles_list)
    if n == 0:
        return sparse.csr_matrix((0, 2 * n_features), dtype=np.float32)

    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1:
        return _transform_chunk(smiles_list, seq_list, n_features, ngram_smiles, ngram_seq).tocsr()

    chunk_size = int(math.ceil(n / n_jobs))
    chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    mats = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_transform_chunk)(
            smiles_list[s:e],
            seq_list[s:e],
            n_features,
            ngram_smiles,
            ngram_seq,
        )
        for (s, e) in chunks
    )
    return sparse.vstack(mats, format="csr")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["davis", "kiba"])
    ap.add_argument("--split", type=str, required=True, choices=["random", "cold_drug", "cold_target", "cold_pair"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="data")

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--solver", type=str, default="lsqr",
                    choices=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"])
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--max_iter", type=int, default=500)

    ap.add_argument("--n_features", type=int, default=2**20)
    ap.add_argument("--ngram_smiles", type=int, default=5)
    ap.add_argument("--ngram_seq", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--exp_name", type=str, default=None)
    args = ap.parse_args()

    exp_name = args.exp_name or f"day4_ridge_{args.dataset}_{args.split}_seed{args.seed}"
    run_dir = make_run_dir("runs", exp_name)

    cfg = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "data_root": args.data_root,
        "model": {
            "name": "ridge_hashing",
            "alpha": args.alpha,
            "solver": args.solver,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "n_features": args.n_features,
            "ngram_smiles": args.ngram_smiles,
            "ngram_seq": args.ngram_seq,
            "n_jobs": args.n_jobs,
        },
    }
    save_yaml(cfg, run_dir / "config.yaml")

    pbar = tqdm(total=9, desc="ridge_baseline", unit="step")
    try:
        ds = PairDataset(root=args.data_root, dataset=args.dataset)
        df = ds.pairs
        pbar.set_postfix_str("load_pairs")
        pbar.update(1)

        idx = load_split_indices(args.data_root, args.dataset, args.split, args.seed)
        d_train = df.iloc[idx["train"]].reset_index(drop=True)
        d_cal = df.iloc[idx["cal"]].reset_index(drop=True)
        d_test = df.iloc[idx["test"]].reset_index(drop=True)
        pbar.set_postfix_str("slice_split")
        pbar.update(1)

        pbar.set_postfix_str("featurize_train")
        X_train = make_features_parallel(d_train["smiles"], d_train["sequence"], args.n_features, args.ngram_smiles, args.ngram_seq, args.n_jobs)
        pbar.update(1)

        pbar.set_postfix_str("featurize_cal")
        X_cal = make_features_parallel(d_cal["smiles"], d_cal["sequence"], args.n_features, args.ngram_smiles, args.ngram_seq, args.n_jobs)
        pbar.update(1)

        pbar.set_postfix_str("featurize_test")
        X_test = make_features_parallel(d_test["smiles"], d_test["sequence"], args.n_features, args.ngram_smiles, args.ngram_seq, args.n_jobs)
        pbar.update(1)

        y_train = d_train["y"].to_numpy(dtype=np.float32)
        y_cal = d_cal["y"].to_numpy(dtype=np.float32)
        y_test = d_test["y"].to_numpy(dtype=np.float32)
        pbar.set_postfix_str("prepare_y")
        pbar.update(1)

        pbar.set_postfix_str("fit_model")
        model = Ridge(alpha=args.alpha, solver=args.solver, tol=args.tol, max_iter=args.max_iter, random_state=args.seed)
        model.fit(X_train, y_train)
        pbar.update(1)

        pbar.set_postfix_str("predict")
        pred_cal = model.predict(X_cal).astype(np.float32)
        pred_test = model.predict(X_test).astype(np.float32)
        pbar.update(1)

        pbar.set_postfix_str("eval_and_save")
        metrics = {
            "dataset": args.dataset,
            "split": args.split,
            "seed": int(args.seed),
            "rmse_cal": rmse(y_cal, pred_cal),
            "rmse_test": rmse(y_test, pred_test),
            "cindex_cal": concordance_index(y_cal, pred_cal),
            "cindex_test": concordance_index(y_test, pred_test),
            "n_train": int(len(d_train)),
            "n_cal": int(len(d_cal)),
            "n_test": int(len(d_test)),
        }

        preds = d_test[["drug_idx", "target_idx", "y"]].copy()
        preds["y_pred"] = pred_test
        preds_path = run_dir / "preds_test.csv.gz"
        preds.to_csv(preds_path, index=False, compression="gzip")

        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (run_dir / "stdout.log").write_text(
            f"Saved preds: {preds_path}\nMetrics:\n{json.dumps(metrics, indent=2)}\n",
            encoding="utf-8",
        )
        pbar.update(1)

        print(json.dumps(metrics, indent=2))
    finally:
        pbar.close()


if __name__ == "__main__":
    main()
