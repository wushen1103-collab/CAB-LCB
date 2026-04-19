# CAS-LCB

Code repository for the paper:

**CAS-LCB: Reliable Top-k Drug-Target Screening under Random and Cold Benchmark Shifts via Selection-Aware Conformal Auto-Selection**

## Overview

CAS-LCB is a selection-aware conformal screening framework for reliable top-k drug-target screening under benchmark shift. The repository follows the paper's one-touch evaluation setting: model fitting, conformal calibration, configuration selection, and final testing are kept separate so that the deployed shortlist can be audited without hidden split reuse.

The public repository is intentionally lightweight. It keeps the core code needed to understand and reproduce the released workflow, while excluding large datasets, run artifacts, mirrored external backbones, and local experiment scratch space.

## Paper-Aligned Method Scope

- `CAS-LCB` selects a candidate conformal pipeline by maximizing a lower confidence bound on shortlist hit quality on an independent selection split.
- `CAS-LCB-Bonferroni` is the more conservative variant used for broader candidate sweeps.
- The repository terminology follows the latest manuscript: `one-touch`, `four-way protocol`, `selection-aware`, and `top-k screening`.

## Repository Layout

- `configs/`: base configuration files for lightweight runs and examples.
- `src/`: small shared library utilities for data I/O, conformal routines, and evaluation helpers.
- `scripts/`: experiment and analysis entry points used in the released workflow.
- `tools/`: packaging helpers for lightweight release bundles.

## Core Scripts

The repository contains many utility scripts collected during the study, but the paper-facing core is centered on:

- `scripts/prepare_dataset.py`: dataset preparation entry point.
- `scripts/make_splits.py`: split generation utilities.
- `scripts/make_calcp_splits.py`: calibration-aware split preparation.
- `scripts/split_test_into_eval_test.py`: one-touch evaluation/test partitioning helper.
- `scripts/train_deepdta_point.py`: DeepDTA point predictor training.
- `scripts/train_graphdta_point.py`: GraphDTA point predictor training.
- `scripts/run_local_conformal_from_preds.py`: local conformal inference from prediction files.
- `scripts/run_cluster_conformal_from_preds.py`: clustered conformal inference from prediction files.
- `scripts/run_knn_conformal_from_preds.py`: kNN-style conformal inference from prediction files.
- `scripts/run_split_conformal_from_preds.py`: split conformal inference from prediction files.
- `scripts/build_constrained_autosel_lcb.py`: selection-aware lower-bound scoring for constrained configuration search.
- `scripts/build_deepdta_constrained_autosel.py`: DeepDTA-oriented CAS-LCB selection assembly.
- `scripts/compute_lcb_at_k.py`: top-k hit and LCB utility computation.
- `scripts/reproduce_lcb_at_k.sh`: lightweight reproduction helper for released audit CSVs.

## What Is Intentionally Excluded

The following content is intentionally not tracked in the public repository:

- raw or processed datasets
- run directories and checkpoints
- logs and temporary outputs
- large result tables and figure exports
- mirrored third-party backbone repositories
- local environment folders and agent scratch directories

This keeps the public repository aligned with the paper while avoiding unnecessary bulk and process-specific clutter.

## Lightweight Release Bundle

To create a compact release bundle from tracked files only, run:

```bash
bash tools/make_release.sh
```

The release helper exports a paper-aligned subset of the repository without pulling in datasets or run artifacts.

## Notes

- Internal module names such as `src/dti_cp` are kept for code continuity.
- Public-facing terminology and method labels follow the latest manuscript, including `CAS-LCB-Bonferroni`.
