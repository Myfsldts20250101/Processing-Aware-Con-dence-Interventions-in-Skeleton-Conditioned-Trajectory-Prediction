# 2026-10-12 绘图更新

本投稿版本四张正文图由 `redraw_main_figures_20261012.py` 生成（PDF、SVG、400 dpi PNG）。原始统计数据未改变。`regenerate_statistical_figures.py` 是旧版绘图程序，保留供历史核验；重绘当前投稿图请使用新程序。具体来源与统计口径见 `style_redraw_metadata_20261012.json`。补充材料六张图沿用原版。

# Statistical figure sources — 2026-10-10

Two statistical figures were regenerated from numerical sources, without image upsampling, training, or model inference.

## Main manuscript figure files

- ../pdf/correspondence_nj_policy_support.pdf — vector PDF, generated from 40 cell means.
- ../pdf/correspondence_u1f_response_deciles.pdf — vector PDF, generated from the archived ten-bin summary of the historical F display sample.
- Matching PNG files were rendered at 400 dpi (2640 × 1540 and 2640 × 1700 pixels).
- Matching editable SVG files are in this directory.

## Paired processing and support comparisons

The 40 local per_window_NJ.csv tables were filtered by include_main=1 (15,160 records per cell). The plot shows, for each of eight training seeds, the mean of five donor-specific window means of delta_policy and delta_support. Whiskers span the five donor means, not confidence intervals.

nj_policy_support_cell_means.csv contains all plotted source means. nj_policy_support_source_manifest.json records the SHA-256 digest of each original table. Recomputed eight-seed mean/sample SD values are:

- Processing: 0.0025818016232578957 / 0.0024718588797761052; seven positive, one negative.
- Assignment support: 0.010544789401983052 / 0.007859615942456764; eight positive.

The geometry and labels were redrawn; the plotted statistics and donor min–max convention are retained.

## Historical F response deciles

historical_F_FIGS_SUMMARY_20261005.json is an unchanged copy of the source summary. Its SHA-256 is 2b467cb631cfc7bb03704b43c23ba454415e3faedc72941b465e09e54d40e716.

The source records ten response bins, each with 8,000 sampled cell-window records. Exact integer category counts were recovered by multiplying archived proportions by 8,000; integer recovery and pooled totals were checked. F_response_deciles_archived_sample_20261005.csv contains those counts and proportions.

Categories at tau=0.01 are improved (delta ADE < -0.01), in band (absolute delta ADE <= 0.01), and worsened (delta ADE > 0.01). The ten bins represent the archived 80,000-record pool (up to 2,000 per cell; archived sample seed 20261005).

The historical individual sample rows and original sampling script are unavailable. Replotting uses the archived ten-bin summary without replacement sampling or later F evaluations. The graph is descriptive: cell-window records are repeated evaluations, and the full-data hierarchical estimates remain unchanged.

## Replotting

For the historical two-figure version, run `regenerate_statistical_figures.py`. With the bundled CSV and archived JSON present, no original per-window data are needed to redraw both figures. NumPy and Matplotlib are required; tested with NumPy 2.4.6 and Matplotlib 3.11.1.

For fresh recomputation of the 40 cell means, use --refresh-cell-means and optionally --nj-source pointing to the original nj_same_assign_v1 directory. The script then checks all 40 cells and their record counts and refreshes the source manifest.

figure_regeneration_metadata.json contains the numerical audit. `matplotlib_cache/` contains a local font cache and is excluded from distribution.
