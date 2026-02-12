from __future__ import annotations

import argparse
import ast
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dti_cp.utils.io import make_run_dir, save_yaml

def _try_load_pickle(path: Path) -> Any | None:
    for kwargs in ({}, {"encoding": "latin1"}):
        try:
            with open(path, "rb") as f:
                return pickle.load(f, **kwargs)
        except Exception:
            continue
    return None

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()

def _parse_text_blob(s: str) -> Any | None:
    if not s:
        return None
    # JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # Python literal (dict/list/...)
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    return None

def _normalize_entity(obj: Any, kind: str) -> list[str]:
    """
    kind: 'ligand' or 'protein'
    """
    if isinstance(obj, list):
        return [str(x) for x in obj]

    if isinstance(obj, dict):
        keys = list(obj.keys())

        def _is_intlike(x: Any) -> bool:
            if isinstance(x, int):
                return True
            if isinstance(x, str) and x.isdigit():
                return True
            return False

        if all(_is_intlike(k) for k in keys):
            keys_sorted = sorted(keys, key=lambda k: int(k))
        else:
            keys_sorted = sorted(keys, key=lambda k: str(k))

        out: list[str] = []
        for k in keys_sorted:
            v = obj[k]
            if isinstance(v, dict):
                if kind == "ligand":
                    for cand in ("smiles", "canonical_smiles", "can_smiles"):
                        if cand in v:
                            out.append(str(v[cand]))
                            break
                    else:
                        out.append(str(v))
                else:
                    for cand in ("sequence", "seq"):
                        if cand in v:
                            out.append(str(v[cand]))
                            break
                    else:
                        out.append(str(v))
            else:
                out.append(str(v))
        return out

    if isinstance(obj, str):
        # fall back to treating as multi-line text
        lines = [x.strip() for x in obj.splitlines() if x.strip()]
        return lines

    return [str(obj)]

def _load_entity_file(path: Path, kind: str) -> list[str]:
    pkl_obj = _try_load_pickle(path)
    if pkl_obj is not None:
        return _normalize_entity(pkl_obj, kind)

    text = _read_text(path)
    parsed = _parse_text_blob(text)
    if parsed is not None:
        return _normalize_entity(parsed, kind)

    # last resort: line-based
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines

def _load_affinity_matrix(raw_dir: Path) -> np.ndarray:
    y_path_candidates = [
        raw_dir / "Y",
        raw_dir / "Y.txt",
        raw_dir / "Y.pkl",
        raw_dir / "Y.pickle",
    ]
    for yp in y_path_candidates:
        if not yp.exists():
            continue
        obj = _try_load_pickle(yp)
        if obj is not None:
            return np.array(obj)
        try:
            return np.loadtxt(yp)
        except Exception:
            continue
    raise FileNotFoundError(f"Could not find/load affinity matrix in {raw_dir}. Tried: {y_path_candidates}")

def build_pairs_from_matrix(ligands: list[str], proteins: list[str], Y: np.ndarray) -> pd.DataFrame:
    if Y.ndim != 2:
        raise ValueError(f"Expected Y to be 2D, got shape={Y.shape}")

    n_drug, n_target = Y.shape
    if n_drug != len(ligands) or n_target != len(proteins):
        raise ValueError(
            f"Shape mismatch: Y={Y.shape}, len(ligands)={len(ligands)}, len(proteins)={len(proteins)}"
        )

    mask = np.isfinite(Y)
    drug_idx, target_idx = np.where(mask)
    y = Y[drug_idx, target_idx].astype(np.float32)

    df = pd.DataFrame(
        {
            "drug_idx": drug_idx.astype(np.int32),
            "target_idx": target_idx.astype(np.int32),
            "y": y,
        }
    )
    df["smiles"] = [ligands[i] for i in df["drug_idx"].to_numpy()]
    df["sequence"] = [proteins[i] for i in df["target_idx"].to_numpy()]
    return df

def write_gz_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["davis", "kiba"])
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--exp_name", type=str, default=None)
    args = ap.parse_args()

    dataset = args.dataset
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    exp_name = args.exp_name or f"prep_{dataset}"
    run_dir = make_run_dir("runs", exp_name)

    cfg = {"dataset": dataset, "raw_dir": str(raw_dir), "out_dir": str(out_dir)}
    save_yaml(cfg, run_dir / "config.yaml")

    lig_path = raw_dir / "ligands_can.txt"
    prot_path = raw_dir / "proteins.txt"

    ligands = _load_entity_file(lig_path, kind="ligand")
    proteins = _load_entity_file(prot_path, kind="protein")
    Y = _load_affinity_matrix(raw_dir)

    pairs = build_pairs_from_matrix(ligands, proteins, Y)

    out_pairs = out_dir / "pairs.csv.gz"
    write_gz_csv(pairs, out_pairs)

    meta = {
        "dataset": dataset,
        "n_drugs": int(len(ligands)),
        "n_targets": int(len(proteins)),
        "n_pairs": int(len(pairs)),
        "y_min": float(np.min(pairs["y"])),
        "y_max": float(np.max(pairs["y"])),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "entities.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (run_dir / "stdout.log").write_text(f"Wrote: {out_pairs}\nMeta: {meta}\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
