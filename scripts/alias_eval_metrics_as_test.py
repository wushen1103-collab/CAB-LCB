#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", required=True)
    ap.add_argument("--metrics_name", default="conformal_metrics.json")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    runs_tag = Path(args.runs_tag)
    metric_files = sorted(runs_tag.rglob(args.metrics_name))

    print(f"[INFO] metrics files found: {len(metric_files)}")
    mode = "WRITE" if args.write else "DRY_RUN"
    print(f"[INFO] mode: {mode}")

    patched = 0
    inserted_keys = 0
    already_has_test = 0

    for mp in metric_files:
        cm = load_json(mp)

        has_eval = ("coverage_eval" in cm) or ("avg_width_eval" in cm)
        has_test = ("coverage_test" in cm) or ("avg_width_test" in cm)

        if has_test:
            already_has_test += 1
            continue
        if not has_eval:
            continue

        before = set(cm.keys())

        if "coverage_eval" in cm and "coverage_test" not in cm:
            cm["coverage_test"] = cm["coverage_eval"]
        if "avg_width_eval" in cm and "avg_width_test" not in cm:
            cm["avg_width_test"] = cm["avg_width_eval"]

        cm["test_metrics_alias_of"] = "eval"

        after = set(cm.keys())
        inserted = len(after - before)
        if inserted > 0:
            patched += 1
            inserted_keys += inserted
            if args.write:
                save_json(mp, cm)

    print(f"[INFO] already had *_test: {already_has_test}")
    print(f"[INFO] patched files: {patched}")
    print(f"[INFO] total inserted keys: {inserted_keys}")

    return


if __name__ == "__main__":
    main()
