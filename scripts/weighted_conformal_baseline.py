from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

def unify_pred_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "y_pred" in df.columns:
        return df
    for alt in ["pred", "yhat"]:
        if alt in df.columns:
            df["y_pred"] = df[alt]
            return df
    raise ValueError(f"No prediction column found. Need one of y_pred/pred/yhat. Got: {list(df.columns)[:30]}")

def infer_need_nm_to_pkd(y: pd.Series) -> bool:
    yv = pd.to_numeric(y, errors="coerce").dropna()
    if len(yv) == 0:
        return False
    return float(yv.median()) > 50.0

def nm_to_pkd(x_nm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x_nm, dtype=float)
    x[x <= 0] = eps
    return 9.0 - np.log10(x)

def maybe_scale(df: pd.DataFrame, scale_mode: str) -> tuple[pd.DataFrame, str, dict]:
    """
    scale_mode:
      - auto: convert if y median suggests nm scale
      - nm_to_pkd: always convert
      - none: never convert
    """
    df = df.copy()
    stats = {"clamped_y": 0, "clamped_pred": 0}
    if scale_mode == "none":
        return df, "none", stats

    if "y" not in df.columns:
        # Cannot decide; keep as-is
        return df, "none", stats

    need = False
    if scale_mode == "nm_to_pkd":
        need = True
    elif scale_mode == "auto":
        need = infer_need_nm_to_pkd(df["y"])
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    if not need:
        return df, "none", stats

    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)

    stats["clamped_y"] = int(np.sum(y <= 0))
    stats["clamped_pred"] = int(np.sum(yp <= 0))

    df["y"] = nm_to_pkd(y)
    df["y_pred"] = nm_to_pkd(yp)
    return df, "nm_to_pkd", stats

def featurize_B(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set B:
      - drug_idx, target_idx (categorical)
      - len(smiles), len(sequence) (numeric)
    """
    out = pd.DataFrame()
    if "drug_idx" not in df.columns or "target_idx" not in df.columns:
        raise ValueError("Missing drug_idx/target_idx columns for feature set B.")
    out["drug_idx"] = df["drug_idx"].astype(str)
    out["target_idx"] = df["target_idx"].astype(str)

    if "smiles" in df.columns:
        out["smiles_len"] = df["smiles"].astype(str).str.len()
    else:
        out["smiles_len"] = 0

    if "sequence" in df.columns:
        out["sequence_len"] = df["sequence"].astype(str).str.len()
    else:
        out["sequence_len"] = 0

    return out

def fit_domain_classifier(X_cal: pd.DataFrame, X_test: pd.DataFrame, C: float = 1.0, seed: int = 0):
    """
    Train a domain classifier: cal=0, test=1.
    Uses scikit-learn if available.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise RuntimeError("scikit-learn is required for this baseline. Please install sklearn.") from e

    X = pd.concat([X_cal, X_test], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(X_cal), dtype=int), np.ones(len(X_test), dtype=int)])

    X = pd.get_dummies(X, columns=["drug_idx", "target_idx"], dummy_na=False)
    clf = LogisticRegression(max_iter=2000, C=C, random_state=seed, n_jobs=1)
    clf.fit(X, y)
    return clf, X.columns.tolist()

def predict_test_prob(clf, columns: list[str], X: pd.DataFrame) -> np.ndarray:
    Xd = pd.get_dummies(X, columns=["drug_idx", "target_idx"], dummy_na=False)
    # Align columns
    for c in columns:
        if c not in Xd.columns:
            Xd[c] = 0
    Xd = Xd[columns]
    p = clf.predict_proba(Xd)[:, 1]
    # Avoid division by zero
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p

def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Weighted quantile: smallest v s.t. cumulative weight >= q * total weight.
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cutoff = q * cw[-1]
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calproxy", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--scale_mode", choices=["auto", "none", "nm_to_pkd"], default="auto")
    ap.add_argument("--w_clip_min", type=float, default=0.05)
    ap.add_argument("--w_clip_max", type=float, default=20.0)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cal = pd.read_csv(args.calproxy)
    test = pd.read_csv(args.test)

    cal = unify_pred_col(cal)
    test = unify_pred_col(test)

    # Scale handling (auto nm->pKd)
    cal, cal_scale, cal_stats = maybe_scale(cal, args.scale_mode)
    test, test_scale, test_stats = maybe_scale(test, args.scale_mode)

    # Features B
    X_cal = featurize_B(cal)
    X_test = featurize_B(test)

    # Domain classifier and weights
    clf, cols = fit_domain_classifier(X_cal, X_test, C=args.C, seed=args.seed)
    p_cal = predict_test_prob(clf, cols, X_cal)

    w = p_cal / (1.0 - p_cal)
    w = np.clip(w, args.w_clip_min, args.w_clip_max)

    # Residuals on calproxy
    if "y" not in cal.columns:
        raise ValueError("calproxy must contain y for conformal calibration.")
    resid = np.abs(cal["y"].to_numpy(dtype=float) - cal["y_pred"].to_numpy(dtype=float))

    qhat = weighted_quantile(resid, w, q=1.0 - args.alpha)

    out = test.copy()
    out["pi_lo"] = out["y_pred"] - qhat
    out["pi_hi"] = out["y_pred"] + qhat
    out["width"] = out["pi_hi"] - out["pi_lo"]

    if "y" in out.columns:
        out["covered"] = ((out["y"] >= out["pi_lo"]) & (out["y"] <= out["pi_hi"])).astype(int)
        cov = float(out["covered"].mean())
        mw = float(out["width"].mean())
        meet = float(cov >= 0.90)
        print(
            f"scale(cal/test)={cal_scale}/{test_scale} "
            f"clamp_y(cal/test)={cal_stats['clamped_y']}/{test_stats['clamped_y']} "
            f"clamp_pred(cal/test)={cal_stats['clamped_pred']}/{test_stats['clamped_pred']} "
            f"qhat={qhat:.6g} coverage={cov:.4f} mean_width={mw:.4f} meet_90={meet:.0f}"
        )
    else:
        print(f"qhat={qhat:.6g}")

    out.to_csv(args.out, index=False, compression="gzip")

if __name__ == "__main__":
    main()
