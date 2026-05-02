import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

#load data from results files
with open("scaling_laws/results/SMALL_output.json") as f:
    small_data = json.load(f)["SMALL"]

with open("scaling_laws/results/MEDIUM_output.json") as f:
    medium_data = json.load(f)["MEDIUM"]

with open("scaling_laws/results/LARGE_output.json") as f:
    large_data = json.load(f)["LARGE"]

#extract flops and val_loss, skip step 0 
def extract(data):
    flops = [d["flops"] for d in data if d["flops"] > 0]
    loss  = [d["val_loss"] for d in data if d["flops"] > 0]
    return np.array(flops), np.array(loss)

flops_small,  loss_small  = extract(small_data)
flops_medium, loss_medium = extract(medium_data)
flops_large,  loss_large  = extract(large_data)

#fit power law in log-log space
def fit_power_law(flops, loss):
    log_f = np.log10(flops)
    log_l = np.log10(loss)
    slope, intercept, r, _, _ = linregress(log_f, log_l)
    f_range = np.linspace(flops.min(), flops.max(), 300)
    l_fit   = 10 ** (intercept + slope * np.log10(f_range))
    return f_range, l_fit, slope, r**2

f_range_s, l_fit_s, slope_s, r2_s = fit_power_law(flops_small,  loss_small)
f_range_m, l_fit_m, slope_m, r2_m = fit_power_law(flops_medium, loss_medium)
f_range_l, l_fit_l, slope_l, r2_l = fit_power_law(flops_large,  loss_large)

#plot 1: log-log loss vs compute
plt.figure(figsize=(8, 5))

plt.scatter(flops_small,  loss_small,  label="SMALL  (~786K params)", color="blue",   s=40)
plt.scatter(flops_medium, loss_medium, label="MEDIUM (~4M params)",   color="orange", s=40)
plt.scatter(flops_large,  loss_large,  label="LARGE  (~25M params)",  color="green",  s=40)

plt.plot(f_range_s, l_fit_s, color="blue",   linewidth=1.5, linestyle="--",
         label=f"SMALL fit  slope={slope_s:.3f} R²={r2_s:.3f}")
plt.plot(f_range_m, l_fit_m, color="orange", linewidth=1.5, linestyle="--",
         label=f"MEDIUM fit slope={slope_m:.3f} R²={r2_m:.3f}")
plt.plot(f_range_l, l_fit_l, color="green",  linewidth=1.5, linestyle="--",
         label=f"LARGE fit  slope={slope_l:.3f} R²={r2_l:.3f}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Compute (FLOPs)")
plt.ylabel("Validation Loss")
plt.title("Loss vs Compute [log-log] — Scaling Laws Experiment")
plt.legend(fontsize=8)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("scaling_laws/results/loss_vs_compute.png", dpi=150)
plt.show()
print("Saved loss_vs_compute.png")

#plot 2: val loss vs training steps
steps_small  = [d["step"] for d in small_data  if d["flops"] > 0]
steps_medium = [d["step"] for d in medium_data if d["flops"] > 0]
steps_large  = [d["step"] for d in large_data  if d["flops"] > 0]

plt.figure(figsize=(8, 5))
plt.plot(steps_small,  loss_small,  marker="o", label="SMALL  (~786K)", color="blue")
plt.plot(steps_medium, loss_medium, marker="o", label="MEDIUM (~4M)",   color="orange")
plt.plot(steps_large,  loss_large,  marker="o", label="LARGE  (~25M)",  color="green")

plt.xlabel("Training Step")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Training Steps")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("scaling_laws/results/loss_vs_steps.png", dpi=150)
plt.show()
print("Saved loss_vs_steps.png")

#print power law summary
print("\n=== Power Law Fit Summary ===")
print(f"SMALL  — slope: {slope_s:.4f}  R²: {r2_s:.4f}")
print(f"MEDIUM — slope: {slope_m:.4f}  R²: {r2_m:.4f}")
print(f"LARGE  — slope: {slope_l:.4f}  R²: {r2_l:.4f}")