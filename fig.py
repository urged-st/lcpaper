import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob

os.makedirs("figures", exist_ok=True)

var_df = pd.read_csv("results/variability_metrics.csv")
snr_df = pd.read_csv("results/detection_snr.csv")
df = pd.merge(var_df, snr_df, on="star")

colors = {"AU Mic": "#e05c5c", "WASP-52": "#f0a500", "HD 209458": "#5c9ee0"}

epochs = {
    "AU Mic":    1330.3915,
    "WASP-52":   3560.1000,
    "HD 209458": 2826.6822,
}

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], hspace=0.45, wspace=0.35)
ax_main = fig.add_subplot(gs[0, :])

for _, row in df.iterrows():
    c = colors.get(row["star"], "grey")
    ax_main.scatter(row["cdpp_ppm"], row["detection_snr"], s=120, color=c, zorder=5, label=row["star"])
    ax_main.errorbar(row["cdpp_ppm"], row["detection_snr"],
                     yerr=row["snr_err"],
                     fmt="none", color=c, capsize=4, linewidth=1.2)
    ax_main.annotate(row["star"], (row["cdpp_ppm"], row["detection_snr"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)

ax_main.set_xlabel("CDPP (ppm)", fontsize=11)
ax_main.set_ylabel("Detection SNR", fontsize=11)
ax_main.set_title("Stellar variability vs transit detection confidence (TESS)", fontsize=12)
ax_main.legend(framealpha=0.5, fontsize=9)

star_order = ["AU Mic", "WASP-52", "HD 209458"]

xlims = {
    "AU Mic":    (-0.5, 0.5),
    "WASP-52":   (-0.5, 0.5),
    "HD 209458": (-0.5, 0.5),
}

for j, name in enumerate(star_order):
    path = f"lightcurves/{name.replace(' ', '_')}.npz"
    ax = fig.add_subplot(gs[1, j])

    if not os.path.exists(path):
        ax.set_title(f"{name}\nno data")
        continue

    d = np.load(path, allow_pickle=True)
    time = d["time"]
    flat_flux = d["flat_flux"]
    period = float(d["period"])
    epoch = epochs[name]

    phase = ((time - epoch) % period) / period
    phase[phase > 0.5] -= 1.0

    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    flux = flat_flux[sort_idx]

    xlim = xlims[name]
    mask = (phase >= xlim[0]) & (phase <= xlim[1])
    phase = phase[mask]
    flux = flux[mask]

    bin_size = 0.01
    bins = np.arange(xlim[0], xlim[1], bin_size)
    valid_bins = [(b, flux[(phase >= b) & (phase < b + bin_size)])
                  for b in bins if np.any((phase >= b) & (phase < b + bin_size))]
    bin_centers = [b + bin_size / 2 for b, _ in valid_bins]
    bin_flux = [f.mean() for _, f in valid_bins]

    ax.scatter(phase, flux, s=0.5, alpha=0.15, color=colors.get(name, "grey"))
    ax.plot(bin_centers, bin_flux, color=colors.get(name, "grey"), linewidth=1.2)
    ax.set_xlim(xlim)
    ax.set_title(f"{name}\nP={period}d", fontsize=8)
    ax.set_xlabel("Phase", fontsize=7)
    ax.set_ylabel("Norm. flux", fontsize=7)
    ax.tick_params(labelsize=7)

    if name == "AU Mic":
        ax.text(0.05, 0.02, "non-detection", transform=ax.transAxes,
                fontsize=7, color="#e05c5c", alpha=0.9)

out = os.path.abspath("figures/variability_vs_snr.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to: {out}")