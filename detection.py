import numpy as np
import pandas as pd
import os
import glob

os.makedirs("results", exist_ok=True)

transit_depths = {
    "AU Mic":    0.0027,
    "WASP-52":   0.0270,
    "HD 209458": 0.0147,
}

records = []

for path in sorted(glob.glob("lightcurves/*.npz")):
    d = np.load(path, allow_pickle=True)
    name = str(d["name"])
    time = d["time"]
    flat_flux = d["flat_flux"]
    period = float(d["period"])
    depth = transit_depths[name]

    phase = (time % period) / period
    phase[phase > 0.5] -= 1.0

    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    flux = flat_flux[sort_idx]

    oot_mask = np.abs(phase) > 0.05
    oot_flux = flux[oot_mask]
    sigma_oot = np.std(oot_flux)
    snr = depth / sigma_oot

    # bootstrap uncertainty on snr
    n_boot = 1000
    boot_snrs = []
    for _ in range(n_boot):
        sample = np.random.choice(oot_flux, size=len(oot_flux), replace=True)
        boot_snrs.append(depth / np.std(sample))
    snr_err = np.std(boot_snrs)

    records.append({
        "star": name,
        "period_d": period,
        "transit_depth": depth,
        "sigma_oot": sigma_oot,
        "detection_snr": snr,
        "snr_err": snr_err,
    })

    print(f"{name}: depth={depth:.4f}, sigma_oot={sigma_oot:.6f}, SNR={snr:.2f} +/- {snr_err:.2f}")

df = pd.DataFrame(records)
df.to_csv("results/detection_snr.csv", index=False)
print("Saved to results/detection_snr.csv")