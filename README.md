# dti-conformal (DTI/DTA + Conformal Prediction)

## Artifact contract (must-have for every run)
Each run writes to: runs/<YYYY-MM-DD>/<exp_name>/
- config.yaml
- metrics.jsonl
- stdout.log
(+ later: preds.parquet/csv, checkpoints, figures)

## Repo layout
- data/        : raw & processed datasets (NOT tracked)
- runs/        : experiment artifacts (NOT tracked)
- results/     : paper-ready tables/figures
- configs/     : configs for experiments
- scripts/     : entry scripts
- src/         : library code
