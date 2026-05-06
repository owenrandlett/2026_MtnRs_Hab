# %%
"""
Plot ZBrain region analysis for three genotype comparisons vs WT as a heatmap.

Reads the *_ZBrain2Analysis.csv files from the zbrain_output folder and
produces a single heatmap:  regions × comparisons.
  Green  = higher pERK in mutant (Mean Positive)
  Magenta = lower pERK in mutant (Mean Negative)

The signed value plotted is: Mean Positive - Mean Negative (linear scale).
All comparisons share the same symmetric colour limits.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = (
    r"Q:\2026_MtnrManuscript\pERKData"
    r"\2023-03-27_pERKtERK_Mtnr1aa+alMutants_60dfwMelatonin"
    r"\output_FDR=5e-05_UsingERK=True\zbrain_output"
)

OUT_DIR = DATA_DIR

# ── which comparisons to show (first group "over" second group = WT) ───────
COMPARISONS = [
    {
        "file": "aa--_al--_over_aa+_al+_SignificantDeltaMedians_ZBrain2Analysis.csv",
        "label": "dKO\n(aa⁻⁻ al⁻⁻) vs WT",
    },
    {
        "file": "aa--_al+_over_aa+_al+_SignificantDeltaMedians_ZBrain2Analysis.csv",
        "label": "aa KO\n(aa⁻⁻ al⁺) vs WT",
    },
    {
        "file": "aa+_al--_over_aa+_al+_SignificantDeltaMedians_ZBrain2Analysis.csv",
        "label": "al KO\n(aa⁺ al⁻⁻) vs WT",
    },
]

# ── colormap: magenta → black → green ──────────────────────────────────────
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "magenta_black_green",
    [
        (0.0, (1.0, 0.0, 1.0)),   # magenta
        (0.5, (0.0, 0.0, 0.0)),   # black
        (1.0, (0.0, 0.7, 0.0)),   # green
    ],
)

# ── load data ──────────────────────────────────────────────────────────────
frames = {}
for comp in COMPARISONS:
    path = os.path.join(DATA_DIR, comp["file"])
    frames[comp["label"]] = pd.read_csv(path, index_col=0)

# ── filter to regions with any signal in any comparison ───────────────────
all_regions = frames[COMPARISONS[0]["label"]].index.tolist()

def has_signal(region):
    for df in frames.values():
        row = df.loc[region]
        if row["Mean Positive"] != 0 or row["Mean Negative"] != 0:
            return True
    return False

active_regions = [r for r in all_regions if has_signal(r)]

# Clean region names: strip leading "-", replace "-" separators with " > "
def clean_name(name):
    return name.strip("-").replace("-", " > ")

col_labels = [comp["label"] for comp in COMPARISONS]

# ── options ────────────────────────────────────────────────────────────────
# Set to an integer to show only the N regions with the largest absolute
# signal (max across all comparisons). Set to None to show all regions.
TOP_N = None

# ── build data matrix (all active regions × comparisons) ─────────────────
# signed value = Mean Positive - Mean Negative
mat_all = np.zeros((len(active_regions), len(COMPARISONS)))
for j, comp in enumerate(COMPARISONS):
    df = frames[comp["label"]]
    for i, region in enumerate(active_regions):
        row = df.loc[region]
        mat_all[i, j] = row["Mean Positive"] - row["Mean Negative"]

# ── filter to top N if requested ─────────────────────────────────────────
if TOP_N is not None:
    row_max = np.max(np.abs(mat_all), axis=1)
    top_idx = np.sort(np.argsort(row_max)[::-1][:TOP_N])  # keep atlas order
    plot_regions = [active_regions[i] for i in top_idx]
    mat = mat_all[top_idx, :]
else:
    plot_regions = active_regions
    mat = mat_all
region_labels = [clean_name(r) for r in plot_regions]

# symmetric colour limits — capped at a percentile so most regions are bright
# raise VMAX_PERCENTILE towards 100 to show more dynamic range;
# lower it to saturate more regions.
VMAX_PERCENTILE = 90
nz = np.abs(mat[mat != 0])
vlim = np.percentile(nz, VMAX_PERCENTILE) if len(nz) > 0 else 1.0

# ── figure ─────────────────────────────────────────────────────────────────
row_height = 0.22   # inches per region
fig_h = max(5, len(plot_regions) * row_height + 2)
fig_w = 2.5 + len(COMPARISONS) * 0.9

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

im = ax.imshow(
    mat,
    aspect="auto",
    cmap=CMAP,
    vmin=-vlim,
    vmax=vlim,
    interpolation="nearest",
)

# axes labels
ax.set_xticks(np.arange(len(COMPARISONS)))
ax.set_xticklabels(col_labels, fontsize=8)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

ax.set_yticks(np.arange(len(plot_regions)))
ax.set_yticklabels(region_labels, fontsize=10)

# thin grid lines between cells
ax.set_xticks(np.arange(len(COMPARISONS)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(active_regions)) - 0.5, minor=True)
ax.grid(which="minor", color="lightgrey", linewidth=0.4)
ax.tick_params(which="minor", length=0)

# colourbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, aspect=40)
cbar.set_label("Δ mean pERK/tERK (mutant − WT)", fontsize=7)
cbar.ax.tick_params(labelsize=7)

fig.suptitle(
    "MAP-Mapping: ZBrain region analysis\nMtnr1aa / Mtnr1al mutants vs WT",
    fontsize=9,
    y=1.01,
)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ZBrain_ThreeComparisons_Heatmap_linear.svg")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")

out_path_pdf = out_path.replace(".svg", ".pdf")
fig.savefig(out_path_pdf, bbox_inches="tight")
print(f"Saved: {out_path_pdf}")

plt.show()
