# Online Resource 2 — Core statistics recomputation package

Version: 1.2 (2026-10-12)

This subpackage recalculates the same-run O / F / N_J / N core comparisons in *Processing-Aware Confidence Interventions in Skeleton-Conditioned Trajectory Prediction*. It is also distributed with the manuscript as `ESM_2.zip`. Recalculation uses the bundled tables and does not require model inference or access to JRDB. Checkpoints and processed inputs are provided elsewhere in the parent diagnostic package. Historical tables and the HST boundary case are outside this subpackage’s scope; public deposition has not been verified.

## Run

Python 3.9 or later; standard library only. Recalculation requires no additional installation, network access or external data.

```text
python reproduce.py
```

The script reads the bundled compressed CSV files, checks their SHA-256 hashes, validates pair identifiers, counts, inclusion flags and frame labels, recalculates the statistics, and compares the entire result with `expected_results.json`. A successful run prints `PASS` and writes `reproduced_results.json` beside the script. Outputs are written to this directory. `--write-expected` is a package-maintainer option for replacing the expected reference; it must not be used to verify an unchanged package.

## Data and scope

- `data/core_windows.csv.gz`: 606,400 rows = eight trained checkpoints (s42–s49) × five donor maps (d42–d46) × 15,160 windows. These are repeated evaluations of a shared window population, **not 606,400 independent samples** and not 40 independent training replications.
- `data/frozen_roster.csv.gz`: the 15,160 target-window keys and their scene IDs. This roster is derived from the included rows, rather than a byte-for-byte copy of an original roster file.
- `data/donor_map_d42.csv.gz` through `data/donor_map_d46.csv.gz`: five mappings reconstructed from the paired rows and verified to be identical across training seeds. Their serialized hashes need not equal hashes of original mapping files with different formats.
- `source_input_hashes.json`: SHA-256 and byte count for the 80 original CSVs plus 40 NJ gate JSONs and 40 cell manifests, with paths relative to the experiment folder.
- `source_join_audit.json`: source-join results and the original NJ gate/cell-manifest contents. Checkpoint and original-roster hashes are provenance recorded by the original run; this package does not ship or revalidate those model files.
- `derived_input_hashes.json`: SHA-256 for the seven package data files actually consumed by the recalculation script.
- `expected_results.json`: complete expected summaries, including all eight seed values and per-cell group sizes.
- `validate.py`: repeats full regression verification and checks selected statistics against separately retained archived summaries. Run `python -B validate.py` from this directory.
- `validation.json`: observed reference differences, software version, input fingerprints and the actual local validation result.
- `references/`: frozen 2026-10-05 aggregate and 2026-10-10 supplementary summaries used for additional reference checks. These are separate archived summaries of the same experiments, not independent experiments or third-party replication.

The complete `expected_results.json` is a frozen regression reference; independent generation of that entire file is not asserted. Final validation does not use `--write-expected` or modify this reference.
The source-extraction utility is retained in the authors' archive. Package-only verification uses the bundled data and does not require original experiment paths.

The scripts recalculate archived results without training, inference or resampling.

## Condition and field dictionary

All ADE values below come from the **same NJ experiment run** and use scaled image-plane coordinates.

| Field | Meaning |
|---|---|
| `train_seed`, `donor_map` | Trained-checkpoint seed and frozen donor-map identifier |
| `scene_id`, `target_key`, `donor_key` | Scene and paired evaluation-window identifiers |
| `include_main` | NJ main-population inclusion flag; all retained rows equal 1 |
| `ade_O`, `ade_F`, `ade_NJ`, `ade_N` | Same-run ground-truth ADE for the four conditions |
| `delta_ade_F`, `delta_ade_NJ`, `delta_ade_N` | F−O, NJ−O and N−O ADE differences, respectively |
| `R_policy`, `R_support` | Same-run prediction response (PredADE) for NJ versus F and N versus NJ |
| `delta_policy`, `delta_support` | Same-run ADE differences NJ−F and N−NJ |
| `n_frame_O`, `n_frame_NJ` | Effective-frame counts recorded in the NJ experiment |
| `n_frame_N_u1_label` | N effective-frame count imported from the matched U1 row, used only to define descriptive frame-count groups |
| `fail_exec`, `nonfinite`, `assign_ok`, `mono_ok` | Per-window execution/finite/assignment/monotonicity checks carried from NJ |
| `u1_include_main`, `u1_n_frame_O_audit_only` | U1 flags/counts retained to recheck population and clean-frame agreement |
| `u1_delta_ade_N_audit_only` | U1 N−O ADE difference retained **only to quantify cross-run numerical disagreement**; never used for substantive effect estimates |

O retains the clean input. F and NJ use the same partial confidence assignment `c_J = where(c_t > 0.1, c_d, c_t)`: F locks clean processing, while NJ recomputes the official processing. N assigns the whole donor confidence field and recomputes the official processing. Confidence values at or below 0.1 were not globally set to zero before assignment. These conditional comparisons do not identify independent causal mechanisms.

## Aggregation and exact sign rules

1. For every checkpoint/donor cell, calculate the window mean for a metric.
2. Average the five donor-cell values within each checkpoint seed.
3. Report the mean and **sample SD (ddof = 1)** of the eight seed means. Positive/negative/zero seed counts refer to those eight means. No independence is asserted between the 40 cells.

For a per-window ADE difference `d`, damage mass is `H = mean(max(d, 0))`; improvement mass is `I = mean(max(-d, 0))`. Thus `H − I = mean(d)`. The script reports H, I, the net difference and the proportions `d > 0`, `d < 0`, `d == 0`, then applies the same donor/seed hierarchy. **No tolerance is applied to sign classification.** Exact zero refers to the stored error difference; it does not establish identical input or identical predicted trajectories.

For frame groups, the label is defined separately by comparing N's U1 frame count with the same-run NJ O count or NJ count. Within each cell and each group, window-weighted summaries average its rows. Scene-equal summaries first average within each scene, then equally average only the scenes containing that group. Five donor summaries are averaged within a seed, followed by the eight-seed mean and sample SD. Group scene sets need not be identical. The N−NJ decreased-frame group has no rows and its effect is undefined.

**Unchanged frame count does not imply identical frame masks, joint masks, valid positions or preprocessing.** These are descriptive post-intervention groups, not randomly assigned groups or estimates of an effect caused by frame-count change. They do not identify joint reactivation.

## Cross-run label join and limits

Original U1 and NJ tables are joined on both target and donor key within the matching seed/donor cell. Every cell contains exactly 15,160 unique, matched pairs with `include_main = 1`; clean-frame counts agree. The maximum absolute discrepancy between the two runs' N−O ADE differences is **approximately 0.0006802529**. The runs therefore differ at the individual-window level. All reported effects use the NJ same-run ADE/difference columns; U1 supplies only the missing N frame-count labels. Matching identifiers and counts alone do not establish tensor-level equivalence across runs.

Verification covers the arithmetic and aggregation of archived core contrasts and frame-count strata in main-paper Tables 2 and 3. Its scope excludes preprocessing validation, distribution shift from donor replacement, training benefits and cross-model or cross-dataset generalization. The reported SD measures variation among eight training-seed summaries on the fixed evaluation population; it is not a sampling-based confidence interval.
