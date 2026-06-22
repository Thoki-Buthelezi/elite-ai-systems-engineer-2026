"""
plot_results.py
---------------
Reads benchmark_results.json and produces two publication-quality
matplotlib figures:

  benchmark_latency.png   — latency (ms) vs sequence length
  benchmark_memory.png    — peak HBM (MB) vs sequence length

Run after benchmark.py:
    python benchmark.py
    python plot_results.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#cccccc",
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.color":         "#e8e8e8",
    "grid.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "axes.titleweight":   "medium",
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "font.family":        "DejaVu Sans",
    "figure.dpi":         150,
})

NAIVE_COLOR  = "#73726c"
TRITON_COLOR = "#185FA5"

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

results_path = Path("benchmark_results.json")
if not results_path.exists():
    print(f"Error: {results_path} not found. Run benchmark.py first.")
    sys.exit(1)

with open(results_path) as f:
    results = json.load(f)

seq_lens   = np.array([r["seq_len"]    for r in results])
lat_naive  = np.array([r["lat_naive"]  for r in results])
lat_triton = np.array([r["lat_triton"] for r in results])
mem_naive  = np.array([r["mem_naive"]  for r in results])
mem_triton = np.array([r["mem_triton"] for r in results])
speedup    = np.array([r["speedup"]    for r in results])
mem_saved  = np.array([r["mem_saved_pct"] for r in results])

cfg = results[0]
config_str = (
    f"batch={cfg['batch']}, heads={cfg['heads']}, "
    f"d_head={cfg['d_head']}"
)

# ---------------------------------------------------------------------------
# Figure 1: Latency
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(
    f"FlashAttention-Triton vs Naive PyTorch — Latency  ({config_str})",
    fontsize=12, fontweight="medium", y=1.01,
)

# Left: raw latency (log scale)
ax = axes[0]
ax.plot(seq_lens, lat_naive,  marker="o", color=NAIVE_COLOR,
        linestyle="--", linewidth=1.8, markersize=5, label="Naive PyTorch")
ax.plot(seq_lens, lat_triton, marker="s", color=TRITON_COLOR,
        linestyle="-",  linewidth=1.8, markersize=5, label="Triton kernel")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlabel("Sequence length")
ax.set_ylabel("Latency (ms, log scale)")
ax.set_title("Raw latency")
ax.legend()

# Right: speedup bar chart
ax2 = axes[1]
bars = ax2.bar(
    np.arange(len(seq_lens)), speedup,
    color=TRITON_COLOR, alpha=0.85, width=0.55,
    zorder=3,
)
ax2.set_xticks(np.arange(len(seq_lens)))
ax2.set_xticklabels([f"{s:,}" for s in seq_lens])
ax2.set_xlabel("Sequence length")
ax2.set_ylabel("Speedup (×)")
ax2.set_title("Triton speedup over naive")
ax2.axhline(1.0, color=NAIVE_COLOR, linestyle="--", linewidth=1, zorder=2)

for bar, val in zip(bars, speedup):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{val:.1f}×",
        ha="center", va="bottom", fontsize=9, color="#333333",
    )

fig.tight_layout()
out_lat = Path("benchmark_latency.png")
fig.savefig(out_lat, bbox_inches="tight")
print(f"Saved {out_lat}")

# ---------------------------------------------------------------------------
# Figure 2: Memory
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
fig2.suptitle(
    f"FlashAttention-Triton vs Naive PyTorch — Peak HBM  ({config_str})",
    fontsize=12, fontweight="medium", y=1.01,
)

# Left: raw memory (log scale)
ax3 = axes2[0]
ax3.plot(seq_lens, mem_naive,  marker="o", color=NAIVE_COLOR,
         linestyle="--", linewidth=1.8, markersize=5, label="Naive PyTorch  O(N²)")
ax3.plot(seq_lens, mem_triton, marker="s", color=TRITON_COLOR,
         linestyle="-",  linewidth=1.8, markersize=5, label="Triton kernel  O(N)")

ax3.set_xscale("log", base=2)
ax3.set_yscale("log")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax3.set_xlabel("Sequence length")
ax3.set_ylabel("Peak HBM (MB, log scale)")
ax3.set_title("Raw peak memory")
ax3.legend()

# Right: memory saved (%) bar chart
ax4 = axes2[1]
bars2 = ax4.bar(
    np.arange(len(seq_lens)), mem_saved,
    color=TRITON_COLOR, alpha=0.85, width=0.55,
    zorder=3,
)
ax4.set_xticks(np.arange(len(seq_lens)))
ax4.set_xticklabels([f"{s:,}" for s in seq_lens])
ax4.set_xlabel("Sequence length")
ax4.set_ylabel("Memory saved (%)")
ax4.set_title("HBM reduction vs naive")
ax4.set_ylim(0, 105)

for bar, val in zip(bars2, mem_saved):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{val:.0f}%",
        ha="center", va="bottom", fontsize=9, color="#333333",
    )

fig2.tight_layout()
out_mem = Path("benchmark_memory.png")
fig2.savefig(out_mem, bbox_inches="tight")
print(f"Saved {out_mem}")

plt.show()