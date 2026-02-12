#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def _run_help(driver: Path) -> str:
    cmd = [sys.executable, str(driver), "-h"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout or ""


def _extract_flags_from_help(help_text: str) -> list[str]:
    flags = []
    for line in help_text.splitlines():
        m = re.match(r"^\s*(--[A-Za-z0-9_\-]+)\b", line)
        if m:
            flags.append(m.group(1))
    return flags


def _pick_runs_flag(help_text: str) -> Optional[str]:
    flags = _extract_flags_from_help(help_text)
    preferred = ["--runs_tag", "--runs-dir", "--runs_dir", "--runs_root", "--runs"]
    for f in preferred:
        if f in flags:
            return f
    for f in flags:
        if "runs" in f.lower():
            return f
    return None


def _pick_eval_test_flag(help_text: str) -> Optional[str]:
    """
    Pick an option flag whose help line suggests it accepts eval/test.
    Works even if the flag name does not contain 'split'.
    """
    candidates = []
    for line in help_text.splitlines():
        if "--" not in line:
            continue
        m = re.match(r"^\s*(--[A-Za-z0-9_\-]+)\b", line)
        if not m:
            continue
        flag = m.group(1)
        low = line.lower()

        score = 0.0
        has_eval = "eval" in low or "{eval" in low
        has_test = "test" in low or "test}" in low
        if has_eval and has_test:
            score += 10.0

        # Still allow split-like names as a weak signal.
        if "split" in flag.lower() or "split" in low:
            score += 2.0

        # Prefer conventional short flags.
        score -= len(flag) * 0.001

        if score > 0:
            candidates.append((score, flag, line))

    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def _count_split_keys(runs_tag: Path, split: str) -> Tuple[int, int]:
    metric_files = sorted(runs_tag.rglob("conformal_metrics.json"))
    want_cov = f"coverage_{split}"
    want_w1 = f"avg_width_{split}"
    want_w2 = f"width_{split}"
    hit = 0
    for p in metric_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (want_cov in obj) or (want_w1 in obj) or (want_w2 in obj):
            hit += 1
    return len(metric_files), hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["eval", "test"])
    ap.add_argument("--driver", type=str, default="scripts/run_split_conformal_from_preds.py")
    ap.add_argument("--extra", type=str, nargs="*", default=[],
                    help="Extra args passed through to the driver.")
    ap.add_argument("--require_write", action="store_true",
                    help="Fail if after running driver no coverage_{split}/(avg_)width_{split} keys exist.")
    args = ap.parse_args()

    runs_tag = Path(args.runs_tag)
    driver = Path(args.driver)

    if not runs_tag.exists():
        raise FileNotFoundError(f"runs_tag not found: {runs_tag}")
    if not driver.exists():
        raise FileNotFoundError(f"driver not found: {driver}")

    help_text = _run_help(driver)
    if not help_text.strip():
        raise RuntimeError("driver -h produced empty output; cannot detect flags")

    runs_flag = _pick_runs_flag(help_text)
    split_like_flag = _pick_eval_test_flag(help_text)

    before_total, before_hit = _count_split_keys(runs_tag, args.split)
    print(f"[INFO] metrics files found: {before_total}")
    print(f"[INFO] before: split='{args.split}' has keys in {before_hit} files")

    cmd = [sys.executable, str(driver)]

    # Pass runs_tag either via a flag if present, otherwise as a positional.
    if runs_flag is not None:
        cmd += [runs_flag, str(runs_tag)]
    else:
        cmd += [str(runs_tag)]

    # Pass eval/test choice if we found a suitable flag; otherwise run driver without it.
    if split_like_flag is not None:
        cmd += [split_like_flag, args.split]
        print(f"[INFO] using split flag: {split_like_flag}")
    else:
        print("[WARN] could not detect an eval/test flag in driver help; running without split flag")

    cmd.extend(args.extra)

    print("[INFO] running driver:")
    print("       " + " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

    after_total, after_hit = _count_split_keys(runs_tag, args.split)
    print(f"[INFO] after: split='{args.split}' has keys in {after_hit} files")

    if args.require_write and after_hit == 0:
        raise SystemExit(
            "[FATAL] Driver finished but no test keys were written.\n"
            "This means the driver did not (or could not) compute split metrics.\n"
            "Next step: inspect driver -h and add/enable export for test metrics, or ensure test preds exist."
        )


if __name__ == "__main__":
    main()
