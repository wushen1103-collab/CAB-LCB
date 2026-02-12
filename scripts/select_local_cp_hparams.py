#!/usr/bin/env python3
"""
Select local conformal hyperparameters using calibration-only split,
then evaluate once on the test set.

This script is designed to work with run_dir containing:
  - preds_cal.csv.gz
  - preds_test.csv.gz

Expected columns (flexible):
  - y column: "y" (preferred) or "affinity"
  - pred column: one of ["pred", "yhat", "y_pred", "prediction", "pred_mean"]
  - smiles column: "smiles" or "SMILES" (required for drug/pair)
  - sequence column: "sequence" or "target_sequence" (required for target/pair)

It writes outputs to:
  run_dir/<out_subdir>/
    - conformal_metrics.json  (test/eval metrics for table collection)
    - selected_params.json    (selection details on cal_select)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
except Exception:
    Chem = None
    AllChem = None
    RDLogger = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_csv_floats(s: str) -> List[float]:
    if s.strip() == "":
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_csv_ints(s: str) -> List[int]:
    if s.strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def float_to_tag(x: float) -> str:
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_preds_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    # pandas can read .csv.gz directly; keep this thin wrapper for clarity
    df = pd.read_csv(path)
    return df


def pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns.tolist())
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive fallback
    low = {c.lower(): c for c in df.columns.tolist()}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def conformal_q_index(n: int, alpha: float) -> int:
    """
    Return the index (0-based) for the conformal quantile using:
      q = ceil((n + 1) * (1 - alpha)) / n
    and taking the corresponding order statistic among n samples.
    """
    if n <= 0:
        return 0
    rank = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(rank, n - 1))
    return rank


def safe_partition_quantile(values_2d: np.ndarray, q_idx: int) -> np.ndarray:
    """
    values_2d: shape (m, k)
    returns: shape (m,)
    """
    if values_2d.ndim != 2:
        raise ValueError("values_2d must be 2D")
    k = values_2d.shape[1]
    if k == 0:
        return np.zeros((values_2d.shape[0],), dtype=np.float64)
    q_idx = max(0, min(q_idx, k - 1))
    part = np.partition(values_2d, q_idx, axis=1)
    return part[:, q_idx].astype(np.float64, copy=False)


def compute_coverage_width(y: np.ndarray, pred: np.ndarray, qhat: np.ndarray) -> Tuple[float, float]:
    err = np.abs(y - pred)
    covered = (err <= qhat).astype(np.float64)
    coverage = float(np.mean(covered)) if covered.size > 0 else float("nan")
    width = float(np.mean(2.0 * qhat)) if qhat.size > 0 else float("nan")
    return coverage, width


def suppress_rdkit_warnings() -> None:
    if RDLogger is None:
        return
    try:
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def morgan_fp(smiles: str, radius: int, nbits: int) -> np.ndarray:
    if Chem is None or AllChem is None:
        raise RuntimeError("RDKit is not available but Morgan fingerprints are requested.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((nbits,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    # RDKit ExplicitBitVect -> numpy
    for i in list(fp.GetOnBits()):
        arr[i] = 1
    return arr.astype(np.float32)


def build_drug_features(
    smiles_list: Sequence[str],
    radius: int,
    nbits: int,
) -> np.ndarray:
    suppress_rdkit_warnings()
    cache: Dict[str, np.ndarray] = {}
    feats = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        if s not in cache:
            cache[s] = morgan_fp(s, radius=radius, nbits=nbits)
        feats[i] = cache[s]
    return feats


def build_target_features_tfidf(
    seq_fit: Sequence[str],
    seq_apply: Sequence[str],
    ngram_min: int,
    ngram_max: int,
    max_features: int,
):
    """
    Fit TF-IDF on seq_fit, transform seq_apply. Return a sparse CSR matrix.

    Keeping the matrix sparse avoids large RAM usage (especially under high parallelism).
    NearestNeighbors(metric='cosine', algorithm='brute') supports sparse CSR input.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        lowercase=False,
        dtype=np.float32,
    )
    X_fit = vectorizer.fit_transform(list(seq_fit))
    X_apply = vectorizer.transform(list(seq_apply))
    # Ensure CSR and float32
    X_apply = X_apply.tocsr().astype(np.float32, copy=False)
    return X_apply


def build_pair_features(
    drug_feats,
    target_feats,
):
    """
    Concatenate drug and target features along feature dimension.
    Supports both dense numpy arrays and sparse CSR matrices.

    If either input is sparse, both are converted to CSR and stacked with scipy.sparse.hstack.
    """
    if drug_feats.shape[0] != target_feats.shape[0]:
        raise ValueError("drug_feats and target_feats must have same number of rows")

    if sparse.issparse(drug_feats) or sparse.issparse(target_feats):
        if not sparse.issparse(drug_feats):
            drug_feats = sparse.csr_matrix(drug_feats, dtype="float32")
        else:
            drug_feats = drug_feats.tocsr().astype("float32", copy=False)

        if not sparse.issparse(target_feats):
            target_feats = sparse.csr_matrix(target_feats, dtype="float32")
        else:
            target_feats = target_feats.tocsr().astype("float32", copy=False)

        X = sparse.hstack([drug_feats, target_feats], format="csr")
        return X.astype("float32", copy=False)

    # Dense fallback
    return np.concatenate([drug_feats, target_feats], axis=1).astype(np.float32, copy=False)


@dataclass
class SelectionResult:
    k_neighbors: int
    min_cal_samples: int
    gamma: float
    coverage_select: float
    width_select: float
    used_global_only: bool


def select_hparams_on_cal(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    pred_fit: np.ndarray,
    X_sel: np.ndarray,
    y_sel: np.ndarray,
    pred_sel: np.ndarray,
    alpha: float,
    k_list: Sequence[int],
    m_list: Sequence[int],
    gamma_list: Sequence[float],
    dist_norm_mode: str = "median",
    knn_metric: str = "cosine",
) -> SelectionResult:
    n_fit = X_fit.shape[0]
    scores_fit = np.abs(y_fit - pred_fit).astype(np.float64, copy=False)

    global_q_idx = conformal_q_index(n_fit, alpha)
    global_qhat = float(np.partition(scores_fit, global_q_idx)[global_q_idx]) if n_fit > 0 else 0.0

    used_global_only = False
    if n_fit == 0:
        used_global_only = True
        qhat = np.full((X_sel.shape[0],), global_qhat, dtype=np.float64)
        cov, wid = compute_coverage_width(y_sel, pred_sel, qhat)
        return SelectionResult(
            k_neighbors=0,
            min_cal_samples=max(m_list) if len(m_list) > 0 else 0,
            gamma=0.0,
            coverage_select=cov,
            width_select=wid,
            used_global_only=True,
        )

    k_max = int(max(k_list)) if len(k_list) > 0 else min(1, n_fit)
    k_max = min(k_max, n_fit)

    nn = NearestNeighbors(metric=knn_metric, algorithm="brute")
    nn.fit(X_fit)
    dists_sel, idx_sel = nn.kneighbors(X_sel, n_neighbors=k_max, return_distance=True)

    # Precompute neighbor scores matrix for fast quantiles
    neighbor_scores_sel = scores_fit[idx_sel]  # shape (n_sel, k_max)

    # Distance normalization reference (calibration-only)
    if dist_norm_mode == "median":
        dist_ref = float(np.median(np.median(dists_sel, axis=1))) if dists_sel.size > 0 else 1.0
        dist_ref = max(dist_ref, 1e-12)
    else:
        dist_ref = 1.0

    best: Optional[SelectionResult] = None

    target_cov = 1.0 - alpha
    for m_min in m_list:
        for k in k_list:
            k_eff = min(int(k), n_fit)
            if k_eff <= 0:
                continue
            q_idx = conformal_q_index(k_eff, alpha)

            qhat_sel = safe_partition_quantile(neighbor_scores_sel[:, :k_eff], q_idx)

            dbar = np.median(dists_sel[:, :k_eff], axis=1).astype(np.float64, copy=False)

            for gamma in gamma_list:
                qhat = qhat_sel
                if float(gamma) > 0.0:
                    inflate = 1.0 + float(gamma) * (dbar / dist_ref)
                    qhat = qhat_sel * inflate

                if n_fit < int(m_min):
                    # global fallback
                    qhat_use = np.full_like(qhat, global_qhat, dtype=np.float64)
                    cov, wid = compute_coverage_width(y_sel, pred_sel, qhat_use)
                    used_global = True
                else:
                    cov, wid = compute_coverage_width(y_sel, pred_sel, qhat)
                    used_global = False

                cand = SelectionResult(
                    k_neighbors=int(k_eff),
                    min_cal_samples=int(m_min),
                    gamma=float(gamma),
                    coverage_select=float(cov),
                    width_select=float(wid),
                    used_global_only=used_global,
                )

                if best is None:
                    best = cand
                    continue

                # Primary: satisfy coverage >= target_cov, then minimize width
                best_ok = best.coverage_select >= target_cov
                cand_ok = cand.coverage_select >= target_cov

                if cand_ok and not best_ok:
                    best = cand
                    continue
                if cand_ok and best_ok:
                    if cand.width_select < best.width_select:
                        best = cand
                    continue
                if (not cand_ok) and (not best_ok):
                    # fallback: maximize coverage, tie-break by width
                    if cand.coverage_select > best.coverage_select:
                        best = cand
                    elif cand.coverage_select == best.coverage_select and cand.width_select < best.width_select:
                        best = cand
                    continue

    assert best is not None
    return best


def eval_on_test(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    alpha: float,
    k_neighbors: int,
    min_cal_samples: int,
    gamma: float,
    knn_metric: str = "cosine",
    dist_norm_mode: str = "median",
) -> Dict[str, float]:
    n_cal = X_cal.shape[0]
    scores_cal = np.abs(y_cal - pred_cal).astype(np.float64, copy=False)

    global_q_idx = conformal_q_index(n_cal, alpha)
    global_qhat = float(np.partition(scores_cal, global_q_idx)[global_q_idx]) if n_cal > 0 else 0.0

    if n_cal == 0:
        qhat = np.full((X_test.shape[0],), global_qhat, dtype=np.float64)
        cov, wid = compute_coverage_width(y_test, pred_test, qhat)
        return {
            "global_qhat": float(global_qhat),
            "coverage_eval": float(cov),
            "avg_width_eval": float(wid),
            "n_eval_used": int(X_test.shape[0]),
            "used_global_only": 1.0,
        }

    k_eff = min(int(k_neighbors), n_cal)
    k_eff = max(1, k_eff)

    nn = NearestNeighbors(metric=knn_metric, algorithm="brute")
    nn.fit(X_cal)

    dists_test, idx_test = nn.kneighbors(X_test, n_neighbors=k_eff, return_distance=True)
    neighbor_scores_test = scores_cal[idx_test]
    q_idx = conformal_q_index(k_eff, alpha)
    qhat_test = safe_partition_quantile(neighbor_scores_test, q_idx)

    # Distance normalization reference computed from calibration only
    if dist_norm_mode == "median":
        # Use calibration self-query to estimate a stable reference scale (exclude self)
        k_ref = min(k_eff + 1, n_cal)
        dists_cal, idx_cal = nn.kneighbors(X_cal, n_neighbors=k_ref, return_distance=True)
        if k_ref >= 2:
            dists_cal_no_self = dists_cal[:, 1:]
        else:
            dists_cal_no_self = dists_cal
        dist_ref = float(np.median(np.median(dists_cal_no_self, axis=1))) if dists_cal_no_self.size > 0 else 1.0
        dist_ref = max(dist_ref, 1e-12)
    else:
        dist_ref = 1.0

    if float(gamma) > 0.0:
        dbar = np.median(dists_test, axis=1).astype(np.float64, copy=False)
        inflate = 1.0 + float(gamma) * (dbar / dist_ref)
        qhat_test = qhat_test * inflate

    if n_cal < int(min_cal_samples):
        qhat_test = np.full_like(qhat_test, global_qhat, dtype=np.float64)
        used_global_only = 1.0
    else:
        used_global_only = 0.0

    cov, wid = compute_coverage_width(y_test, pred_test, qhat_test)
    return {
        "global_qhat": float(global_qhat),
        "coverage_eval": float(cov),
        "avg_width_eval": float(wid),
        "n_eval_used": int(X_test.shape[0]),
        "used_global_only": float(used_global_only),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration-only hyperparam selection for local conformal prediction")
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--alphas", default="0.1", type=str, help="Comma-separated list of alphas, e.g., 0.05,0.1,0.2")

    ap.add_argument("--group_by", default="target", type=str, choices=["target", "drug", "pair"])
    ap.add_argument("--drug_repr", default="morgan", type=str, choices=["morgan"])
    ap.add_argument("--target_repr", default="tfidf", type=str, choices=["tfidf"])

    ap.add_argument("--tfidf_ngram_min", default=3, type=int)
    ap.add_argument("--tfidf_ngram_max", default=5, type=int)
    ap.add_argument("--tfidf_max_features", default=8000, type=int)

    ap.add_argument("--morgan_radius", default=2, type=int)
    ap.add_argument("--morgan_nbits", default=2048, type=int)

    ap.add_argument("--k_list", default="30,60,120,240", type=str)
    ap.add_argument("--m_list", default="50,100,200,400", type=str)
    ap.add_argument("--gamma_list", default="0.0,0.05,0.1", type=str)

    ap.add_argument("--cal_select_frac", default=0.5, type=float)
    ap.add_argument("--select_seed", default=-1, type=int, help="If -1, use run seed if available, else 0")

    ap.add_argument("--knn_metric", default="cosine", type=str)
    ap.add_argument("--dist_norm", default="median", type=str, choices=["median", "none"])

    ap.add_argument("--out_prefix", default="cp_local_autosel", type=str)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    preds_cal_path = run_dir / "preds_cal.csv.gz"
    preds_test_path = run_dir / "preds_test.csv.gz"

    df_cal = read_preds_csv(preds_cal_path)
    df_test = read_preds_csv(preds_test_path)

    y_col = pick_col(df_cal, ["y", "affinity"])
    pred_col = pick_col(df_cal, ["pred", "yhat", "y_pred", "prediction", "pred_mean"])
    smiles_col = pick_col(df_cal, ["smiles", "SMILES"])
    seq_col = pick_col(df_cal, ["sequence", "target_sequence", "seq"])

    if y_col is None:
        raise RuntimeError(f"Cannot locate label column in {preds_cal_path} (expected y/affinity)")
    if pred_col is None:
        raise RuntimeError(f"Cannot locate prediction column in {preds_cal_path} (expected pred/yhat/...)")

    # Determine selection seed
    select_seed = int(args.select_seed)
    if select_seed < 0:
        # try parse from run_dir name "..._seedX"
        m = re.search(r"_seed(\d+)", str(run_dir))
        select_seed = int(m.group(1)) if m else 0
    set_seed(select_seed)

    # Extract arrays
    y_cal = df_cal[y_col].to_numpy(dtype=np.float64)
    pred_cal = df_cal[pred_col].to_numpy(dtype=np.float64)

    y_test = None
    pred_test = None
    y_col_t = pick_col(df_test, ["y", "affinity"])
    pred_col_t = pick_col(df_test, ["pred", "yhat", "y_pred", "prediction", "pred_mean"])
    if y_col_t is None or pred_col_t is None:
        raise RuntimeError("Cannot locate y/pred columns in preds_test.csv.gz")
    y_test = df_test[y_col_t].to_numpy(dtype=np.float64)
    pred_test = df_test[pred_col_t].to_numpy(dtype=np.float64)

    if args.group_by in ("drug", "pair"):
        if smiles_col is None:
            raise RuntimeError("group_by=drug/pair requires smiles column in preds_cal.csv.gz")
    if args.group_by in ("target", "pair"):
        if seq_col is None:
            raise RuntimeError("group_by=target/pair requires sequence column in preds_cal.csv.gz")

    smiles_cal = df_cal[smiles_col].astype(str).tolist() if smiles_col is not None else []
    seq_cal = df_cal[seq_col].astype(str).tolist() if seq_col is not None else []
    smiles_test = df_test[pick_col(df_test, ["smiles", "SMILES"])].astype(str).tolist() if args.group_by in ("drug", "pair") else []
    seq_test = df_test[pick_col(df_test, ["sequence", "target_sequence", "seq"])].astype(str).tolist() if args.group_by in ("target", "pair") else []

    # Split calibration into fit/select
    n_cal = len(y_cal)
    idx = np.arange(n_cal, dtype=np.int64)
    np.random.shuffle(idx)

    frac = float(args.cal_select_frac)
    frac = max(0.1, min(frac, 0.9))
    n_sel = int(round(n_cal * frac))
    n_sel = max(1, min(n_sel, n_cal - 1))
    sel_idx = idx[:n_sel]
    fit_idx = idx[n_sel:]

    y_fit = y_cal[fit_idx]
    pred_fit = pred_cal[fit_idx]
    y_sel = y_cal[sel_idx]
    pred_sel = pred_cal[sel_idx]

    # Build features for selection: fit representations on fit only (for tfidf)
    if args.group_by == "target":
        X_fit = build_target_features_tfidf(
            seq_fit=[seq_cal[i] for i in fit_idx],
            seq_apply=[seq_cal[i] for i in fit_idx],
            ngram_min=args.tfidf_ngram_min,
            ngram_max=args.tfidf_ngram_max,
            max_features=args.tfidf_max_features,
        )
        X_sel = build_target_features_tfidf(
            seq_fit=[seq_cal[i] for i in fit_idx],
            seq_apply=[seq_cal[i] for i in sel_idx],
            ngram_min=args.tfidf_ngram_min,
            ngram_max=args.tfidf_ngram_max,
            max_features=args.tfidf_max_features,
        )
    elif args.group_by == "drug":
        X_fit = build_drug_features(
            smiles_list=[smiles_cal[i] for i in fit_idx],
            radius=args.morgan_radius,
            nbits=args.morgan_nbits,
        )
        X_sel = build_drug_features(
            smiles_list=[smiles_cal[i] for i in sel_idx],
            radius=args.morgan_radius,
            nbits=args.morgan_nbits,
        )
    else:
        # pair
        drug_fit = build_drug_features(
            smiles_list=[smiles_cal[i] for i in fit_idx],
            radius=args.morgan_radius,
            nbits=args.morgan_nbits,
        )
        drug_sel = build_drug_features(
            smiles_list=[smiles_cal[i] for i in sel_idx],
            radius=args.morgan_radius,
            nbits=args.morgan_nbits,
        )
        tgt_fit = build_target_features_tfidf(
            seq_fit=[seq_cal[i] for i in fit_idx],
            seq_apply=[seq_cal[i] for i in fit_idx],
            ngram_min=args.tfidf_ngram_min,
            ngram_max=args.tfidf_ngram_max,
            max_features=args.tfidf_max_features,
        )
        tgt_sel = build_target_features_tfidf(
            seq_fit=[seq_cal[i] for i in fit_idx],
            seq_apply=[seq_cal[i] for i in sel_idx],
            ngram_min=args.tfidf_ngram_min,
            ngram_max=args.tfidf_ngram_max,
            max_features=args.tfidf_max_features,
        )
        X_fit = build_pair_features(drug_fit, tgt_fit)
        X_sel = build_pair_features(drug_sel, tgt_sel)

    alphas = parse_csv_floats(args.alphas)
    k_list = parse_csv_ints(args.k_list)
    m_list = parse_csv_ints(args.m_list)
    gamma_list = parse_csv_floats(args.gamma_list)

    if len(alphas) == 0:
        raise ValueError("alphas cannot be empty")
    if len(k_list) == 0:
        raise ValueError("k_list cannot be empty")
    if len(m_list) == 0:
        raise ValueError("m_list cannot be empty")
    if len(gamma_list) == 0:
        raise ValueError("gamma_list cannot be empty")

    # Final evaluation needs full-cal representations (tfidf fit on full cal)
    for alpha in alphas:
        alpha_tag = float_to_tag(alpha)
        dist_norm_mode = "median" if args.dist_norm == "median" else "none"

        sel = select_hparams_on_cal(
            X_fit=X_fit,
            y_fit=y_fit,
            pred_fit=pred_fit,
            X_sel=X_sel,
            y_sel=y_sel,
            pred_sel=pred_sel,
            alpha=float(alpha),
            k_list=k_list,
            m_list=m_list,
            gamma_list=gamma_list,
            dist_norm_mode=dist_norm_mode,
            knn_metric=args.knn_metric,
        )

        gamma_tag = float_to_tag(sel.gamma)

        out_subdir = f"{args.out_prefix}_{args.group_by}_k{sel.k_neighbors}_m{sel.min_cal_samples}_gamma{gamma_tag}_alpha{alpha_tag}"
        out_dir = run_dir / out_subdir
        if out_dir.exists() and (not args.overwrite):
            print(f"[SKIP] exists: {out_dir}")
            continue
        ensure_dir(out_dir)

        # Build full-cal features for final evaluation
        if args.group_by == "target":
            X_cal_full = build_target_features_tfidf(
                seq_fit=seq_cal,
                seq_apply=seq_cal,
                ngram_min=args.tfidf_ngram_min,
                ngram_max=args.tfidf_ngram_max,
                max_features=args.tfidf_max_features,
            )
            X_test_full = build_target_features_tfidf(
                seq_fit=seq_cal,
                seq_apply=seq_test,
                ngram_min=args.tfidf_ngram_min,
                ngram_max=args.tfidf_ngram_max,
                max_features=args.tfidf_max_features,
            )
        elif args.group_by == "drug":
            X_cal_full = build_drug_features(
                smiles_list=smiles_cal,
                radius=args.morgan_radius,
                nbits=args.morgan_nbits,
            )
            X_test_full = build_drug_features(
                smiles_list=smiles_test,
                radius=args.morgan_radius,
                nbits=args.morgan_nbits,
            )
        else:
            drug_cal = build_drug_features(smiles_cal, radius=args.morgan_radius, nbits=args.morgan_nbits)
            drug_test = build_drug_features(smiles_test, radius=args.morgan_radius, nbits=args.morgan_nbits)
            tgt_cal = build_target_features_tfidf(
                seq_fit=seq_cal,
                seq_apply=seq_cal,
                ngram_min=args.tfidf_ngram_min,
                ngram_max=args.tfidf_ngram_max,
                max_features=args.tfidf_max_features,
            )
            tgt_test = build_target_features_tfidf(
                seq_fit=seq_cal,
                seq_apply=seq_test,
                ngram_min=args.tfidf_ngram_min,
                ngram_max=args.tfidf_ngram_max,
                max_features=args.tfidf_max_features,
            )
            X_cal_full = build_pair_features(drug_cal, tgt_cal)
            X_test_full = build_pair_features(drug_test, tgt_test)

        metrics = eval_on_test(
            X_cal=X_cal_full,
            y_cal=y_cal,
            pred_cal=pred_cal,
            X_test=X_test_full,
            y_test=y_test,
            pred_test=pred_test,
            alpha=float(alpha),
            k_neighbors=int(sel.k_neighbors),
            min_cal_samples=int(sel.min_cal_samples),
            gamma=float(sel.gamma),
            knn_metric=args.knn_metric,
            dist_norm_mode=dist_norm_mode,
        )

        selected_payload = {
            "alpha": float(alpha),
            "group_by": args.group_by,
            "selection_seed": int(select_seed),
            "cal_select_frac": float(frac),
            "k_list": [int(x) for x in k_list],
            "m_list": [int(x) for x in m_list],
            "gamma_list": [float(x) for x in gamma_list],
            "selected": {
                "k_neighbors": int(sel.k_neighbors),
                "min_cal_samples": int(sel.min_cal_samples),
                "distance_inflate_gamma": float(sel.gamma),
                "coverage_select": float(sel.coverage_select),
                "avg_width_select": float(sel.width_select),
                "used_global_only_select": bool(sel.used_global_only),
            },
            "tfidf": {
                "ngram_min": int(args.tfidf_ngram_min),
                "ngram_max": int(args.tfidf_ngram_max),
                "max_features": int(args.tfidf_max_features),
            },
            "morgan": {
                "radius": int(args.morgan_radius),
                "nbits": int(args.morgan_nbits),
            },
        }

        with open(out_dir / "selected_params.json", "w", encoding="utf-8") as f:
            json.dump(selected_payload, f, indent=2, sort_keys=True)

        conformal_payload = {
            "alpha": float(alpha),
            "method": "local",
            "group_by": args.group_by,
            "k_neighbors": int(sel.k_neighbors),
            "min_cal_samples": int(sel.min_cal_samples),
            "distance_inflate_gamma": float(sel.gamma),
            "knn_metric": args.knn_metric,
            "dist_norm": args.dist_norm,
            "global_qhat": float(metrics["global_qhat"]),
            "coverage_eval": float(metrics["coverage_eval"]),
            "avg_width_eval": float(metrics["avg_width_eval"]),
            "n_eval_used": int(metrics["n_eval_used"]),
            "used_global_only": float(metrics["used_global_only"]),
        }

        with open(out_dir / "conformal_metrics.json", "w", encoding="utf-8") as f:
            json.dump(conformal_payload, f, indent=2, sort_keys=True)

        print(f"[OK] wrote: {out_dir}")

    print("[DONE]")


if __name__ == "__main__":
    main()
