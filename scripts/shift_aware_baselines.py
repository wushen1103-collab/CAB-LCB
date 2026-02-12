from __future__ import annotations
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd

def require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Found: {list(df.columns)[:30]} ...")

def unify_pred_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify prediction column name to 'y_pred'.
    Accepts: y_pred, pred, yhat (first match wins).
    """
    df = df.copy()
    if "y_pred" in df.columns:
        return df
    for alt in ["pred", "yhat"]:
        if alt in df.columns:
            df["y_pred"] = df[alt]
            return df
    raise ValueError(
        f"Prediction column not found. Expected one of: y_pred/pred/yhat. Found: {list(df.columns)[:30]}"
    )

def nm_to_pkd(x_nm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert Kd in nM to pKd: pKd = 9 - log10(Kd_nM).
    Clamp non-positive values to eps before log transformation.
    """
    x_nm = np.asarray(x_nm, dtype=float)
    x_nm[x_nm <= 0] = eps  # Clamp non-positive values to eps
    return 9.0 - np.log10(x_nm)

def infer_need_nm_to_pkd(df: pd.DataFrame) -> bool:
    """
    Heuristic: if y median is large, it is likely in nM scale.
    pKd typically ~ [4, 12]. nM often >> 50.
    """
    if "y" not in df.columns:
        return False
    y = pd.to_numeric(df["y"], errors="coerce").dropna()
    if len(y) == 0:
        return False
    return float(y.median()) > 50.0

def maybe_convert_scale(df: pd.DataFrame, do_nm_to_pkd: bool) -> pd.DataFrame:
    """
    Convert both y and y_pred to pKd if needed.
    """
    if not do_nm_to_pkd:
        return df
    df = df.copy()
    if "y" in df.columns:
        df["y"] = nm_to_pkd(pd.to_numeric(df["y"], errors="coerce").to_numpy())
    df["y_pred"] = nm_to_pkd(pd.to_numeric(df["y_pred"], errors="coerce").to_numpy())
    return df

def make_intervals(test_df: pd.DataFrame, qhat: float) -> pd.DataFrame:
    out = test_df.copy()
    out["pi_lo"] = out["y_pred"] - qhat
    out["pi_hi"] = out["y_pred"] + qhat
    if "y" in out.columns:
        out["covered"] = ((out["y"] >= out["pi_lo"]) & (out["y"] <= out["pi_hi"])).astype(int)
    out["width"] = out["pi_hi"] - out["pi_lo"]
    return out

def target_calibrated_split_conformal(
    target_proxy: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
    scale_mode: str
) -> Tuple[pd.DataFrame, float, str]:
    target_proxy = unify_pred_col(target_proxy)
    test = unify_pred_col(test)

    require_cols(target_proxy, ["y", "y_pred"], "target_proxy")
    require_cols(test, ["y_pred"], "test")

    if scale_mode == "auto":
        need = infer_need_nm_to_pkd(target_proxy)
    elif scale_mode == "nm_to_pkd":
        need = True
    elif scale_mode == "none":
        need = False
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    target_proxy = maybe_convert_scale(target_proxy, need)
    test = maybe_convert_scale(test, need)

    resid = np.abs(target_proxy["y"].to_numpy() - target_proxy["y_pred"].to_numpy())
    qhat = float(np.quantile(resid, 1.0 - alpha, method="higher"))
    return make_intervals(test, qhat), qhat, ("nm_to_pkd" if need else "none")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tc_sc"], required=True)
    ap.add_argument("--target_proxy", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--scale_mode", choices=["auto", "none", "nm_to_pkd"], default="auto")
    args = ap.parse_args()

    proxy = pd.read_csv(args.target_proxy)
    test = pd.read_csv(args.test)

    intervals, qhat, applied = target_calibrated_split_conformal(
        proxy, test, args.alpha, args.scale_mode
    )

    if "covered" in intervals.columns:
        cov = float(intervals["covered"].mean())
        w = float(intervals["width"].mean())
        print(f"scale={applied} qhat={qhat:.6g} coverage={cov:.4f} mean_width={w:.4f}")
    else:
        print(f"scale={applied} qhat={qhat:.6g}")

    intervals.to_csv(args.out, index=False, compression="gzip")

if __name__ == "__main__":
    main()
