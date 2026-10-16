"""Regenerate vector statistics figures from archived summaries.
Run beside the bundled CSV and JSON. Raw NJ CSVs are read only when the
bundled cell-means CSV is absent, or when --refresh-cell-means is requested.
"""
from pathlib import Path
import os, json, csv, hashlib, argparse
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PDF = OUT / "pdf"
ROOT = OUT.parent.parent
PDF.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(HERE / "matplotlib_cache")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
parser = argparse.ArgumentParser()
parser.add_argument("--nj-source", type=Path, default=ROOT/"输出"/"donor confiendce"/"nj_same_assign_v1")
parser.add_argument("--refresh-cell-means", action="store_true")
args = parser.parse_args()
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
seeds = list(range(42, 50))
donors = list(range(42, 47))
cell_path = HERE / "nj_policy_support_cell_means.csv"
source_manifest_path = HERE / "nj_policy_support_source_manifest.json"
if not cell_path.exists() or args.refresh_cell_means:
    cells, manifests = [], []
    for seed in seeds:
        for donor in donors:
            p = args.nj_source / "by_ckpt" / f"s{seed}" / f"d{donor}" / "per_window_NJ.csv"
            blob = p.read_bytes()
            policies, supports = [], []
            with p.open(newline="", encoding="utf-8-sig") as handle:
                for r in csv.DictReader(handle):
                    if int(r["include_main"]) == 1:
                        policies.append(float(r["delta_policy"]))
                        supports.append(float(r["delta_support"]))
            assert len(policies) == 15160, (p, len(policies))
            dp, ds = np.array(policies), np.array(supports)
            assert np.isfinite(np.r_[dp, ds]).all()
            cells.append({"train_seed": seed, "donor_map": donor, "n": len(dp),
                          "delta_policy": float(dp.mean()), "delta_support": float(ds.mean())})
            manifests.append({"source": str(p.relative_to(args.nj_source)),
                              "sha256": hashlib.sha256(blob).hexdigest(), "n_include_main": len(dp)})
    with cell_path.open("w", newline="", encoding="utf-8") as handle:
        w=csv.DictWriter(handle, fieldnames=list(cells[0]))
        w.writeheader(); w.writerows(cells)
    source_manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
else:
    with cell_path.open(newline="", encoding="utf-8") as handle:
        cells = list(csv.DictReader(handle))
    for row in cells:
        for k in ("train_seed","donor_map","n"): row[k] = int(row[k])
        for k in ("delta_policy","delta_support"): row[k] = float(row[k])
assert len(cells) == 40
assert {(c["train_seed"],c["donor_map"]) for c in cells} == {(s,d) for s in seeds for d in donors}
assert all(c["n"] == 15160 for c in cells)
blue, orange = "#24628B", "#B85C38"
fig, ax = plt.subplots(figsize=(6.6, 3.85), constrained_layout=True)
stats = {}
for metric, offset, color, mark, label in [
    ("delta_policy", -0.085, blue, "o", r"Processing: $N_J-F$"),
    ("delta_support", 0.085, orange, "s", r"Assignment support: $N-N_J$")]:
    vals = np.array([[c[metric] for c in cells if c["train_seed"] == s] for s in seeds])
    means = vals.mean(axis=1)
    ax.errorbar(np.array(seeds)+offset, means,
                yerr=np.vstack([means-vals.min(axis=1), vals.max(axis=1)-means]),
                color=color, marker=mark, markersize=5, linestyle="none",
                elinewidth=1.1, capsize=3, capthick=1.0, label=label)
    stats[metric] = {"mean": float(means.mean()), "sample_sd": float(means.std(ddof=1)),
                     "positive_seeds": int((means>0).sum()), "negative_seeds": int((means<0).sum()),
                     "per_seed_mean": {str(s): float(v) for s,v in zip(seeds,means)}}
ax.axhline(0, color="#657079", lw=0.9, ls="--")
ax.set_xticks(seeds)
ax.set_xlabel("Training seed")
ax.set_ylabel("Mean GT ADE difference")
ax.set_title("Paired error changes across training seeds", loc="left", fontweight="bold", pad=10)
ax.yaxis.grid(True, color="#DDE2E6", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=8)
for ext in ("pdf", "svg"):
    dest = PDF if ext=="pdf" else HERE
    fig.savefig(dest / f"correspondence_nj_policy_support.{ext}")
fig.savefig(PDF / "correspondence_nj_policy_support.png", dpi=400)
plt.close(fig)

hist_path = HERE / "historical_F_FIGS_SUMMARY_20261005.json"
hist_bytes = hist_path.read_bytes()
hist = json.loads(hist_bytes)
bins = hist["response_bins"]
n = np.array(bins["counts"],dtype=int)
assert len(n) == 10 and (n == 8000).all() and n.sum() == 80000
improve=np.array(bins["improve"])
in_band=np.array(bins["stable"])
worsen=np.array(bins["worsen"])
assert np.allclose(improve+in_band+worsen, 1, rtol=0,atol=1e-14)
assert all(np.max(np.abs(v*n-np.rint(v*n))) < 1e-9 for v in (improve,in_band,worsen))
decile_rows = []
for i in range(10):
    decile_rows.append({
        "response_decile":i+1, "n_sampled_cell_window_records":int(n[i]),
        "improved_count":int(round(improve[i]*n[i])),
        "in_band_count":int(round(in_band[i]*n[i])),
        "worsened_count":int(round(worsen[i]*n[i])),
        "improved_fraction":float(improve[i]),
        "in_band_fraction":float(in_band[i]),
        "worsened_fraction":float(worsen[i])})
with (HERE/"F_response_deciles_archived_sample_20261005.csv").open("w",newline="",encoding="utf-8") as handle:
    w=csv.DictWriter(handle,fieldnames=list(decile_rows[0]));w.writeheader();w.writerows(decile_rows)
pooled = {"improve":float(np.average(improve,weights=n)),
          "stable":float(np.average(in_band,weights=n)),
          "worsen":float(np.average(worsen,weights=n))}
assert all(abs(pooled[k]-hist["pool_band_frac_tau_main"][k]) < 1e-14 for k in pooled)
fig, ax = plt.subplots(figsize=(6.6, 4.25), constrained_layout=True)
xs = np.arange(1,11)
ax.bar(xs, improve, color="#23876F", label=r"Improved: $\Delta$ADE < -0.01", width=.78)
ax.bar(xs, in_band, bottom=improve, color="#D8E0E7", label=r"In band: $|\Delta$ADE| $\leq$ 0.01", width=.78)
ax.bar(xs, worsen, bottom=improve+in_band, color="#B85C38", label=r"Worsened: $\Delta$ADE > +0.01", width=.78)
ax.set_xticks(xs)
ax.set_ylim(0,1)
ax.set_yticks(np.linspace(0,1,6))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel(r"Output-response decile of $\mathrm{Pred\!-\!ADE}_F$ (low to high)")
ax.set_ylabel("Fraction of sampled cell-window records")
ax.set_title("Error direction across output-response deciles", loc="left", fontweight="bold", pad=43)
ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(.5,1.015),frameon=False,
          fontsize=7.0,handlelength=1.0,handletextpad=.4,columnspacing=.9)
ax.set_axisbelow(True)
ax.yaxis.grid(True,color="#EFF2F4",lw=.5)
for ext in ("pdf","svg"):
    dest = PDF if ext=="pdf" else HERE
    fig.savefig(dest / f"correspondence_u1f_response_deciles.{ext}")
fig.savefig(PDF / "correspondence_u1f_response_deciles.png",dpi=400)
plt.close(fig)

metadata = {
 "created":"2026-10-10", "no_training_or_inference":True,
 "policy_support":{
     "n_source_records":606400, "n_cells":40,
     "point":"Mean of five donor cell-window means per training seed",
     "whisker":"Minimum and maximum of five donor cell means; not confidence intervals",
     "cell_means_csv_sha256":hashlib.sha256(cell_path.read_bytes()).hexdigest(),
     "summary":stats},
 "F_response_deciles":{
     "source":"Archived FIGS_SUMMARY.json from 2000窗 sample-seed=20261005",
     "source_sha256":hashlib.sha256(hist_bytes).hexdigest(),
     "archived_source_meta":hist["meta"], "n_deciles":10,
     "n_records_per_decile":8000, "n_plot_records_represented":80000,
     "counts_recovered_from":"Exact archived proportions times archived integer bin counts; checked integer recovery and pooled fractions",
     "pooled_fractions":pooled,
     "historical_sample_limit":"Plot is redrawn from archived decile summaries. Original 80000-row sample and sampling code were not available locally. No claim that individual historical sampled records have been recovered.",
     "inference_limit":"Descriptive pooled cell-window records, not independent observations. Main tables retain full-data hierarchical statistics.",
     "unchanged_full_hierarchy":True},
 "outputs":{
     "vector_pdf":["correspondence_nj_policy_support.pdf","correspondence_u1f_response_deciles.pdf"],
     "png":{"dpi":400,"policy_support_pixels":[2640,1540],"response_deciles_pixels":[2640,1700]},
     "editable":"Python, vector SVG, 40-cell means CSV, archived summary JSON, exact 10-bin counts CSV"},
 "software":{"numpy":np.__version__,"matplotlib":matplotlib.__version__}}
(HERE/"figure_regeneration_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(metadata,ensure_ascii=False,indent=2))
