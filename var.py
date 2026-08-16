import numpy as np
import pandas as pd
import os
import glob

os.makedirs("results", exist_ok=True)

# transit durations as fraction of period for masking (approximate)
transit_durations = {
    "AU Mic":    0.05,
    "WASP-52":   0.05,
    "HD 209458": 0.05,
}

records = []

for path in sorted(glob.glob("lightcurves/*.npz")):
    d = np.load(path, allow_pickle=True)
    name = str(d["name"])
    time = d["time"]
    flat_flux = d["flat_flux"]
    period = float(d["period"])
    dur_frac = transit_durations.get(name, 0.05)

    phase = (time % period) / period
    phase[phase > 0.5] -= 1.0
    oot_mask = np.abs(phase) > dur_frac
    flux_oot = flat_flux[oot_mask]

    rms = np.sqrt(np.mean((flux_oot - np.mean(flux_oot)) ** 2))
    peak_to_peak = np.max(flux_oot) - np.min(flux_oot)

    bin_size = 60
    n_bins = len(flux_oot) // bin_size
    binned = flux_oot[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    cdpp_ppm = np.std(binned) * 1e6

    records.append({
        "star": name,
        "rms": rms,
        "cdpp_ppm": cdpp_ppm,
        "peak_to_peak": peak_to_peak,
    })

    print(f"{name}: RMS={rms:.6f}, CDPP={cdpp_ppm:.1f}ppm, P2P={peak_to_peak:.6f}")

df = pd.DataFrame(records)
df.to_csv("results/variability_metrics.csv", index=False)
print("Saved to results/variability_metrics.csv")