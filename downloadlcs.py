import lightkurve as lk
import numpy as np
import os

stars = [
    ("AU Mic",    "AU Mic",    8.463,  True),
    ("WASP-52",   "WASP-52",   1.7497, False),
    ("HD 209458", "HD 209458", 3.5247, False),
]

os.makedirs("lightcurves", exist_ok=True)

for name, search_term, period, stitch in stars:
    out_path = f"lightcurves/{name.replace(' ', '_')}.npz"
    if os.path.exists(out_path):
        print(f"{name}: already downloaded, skipping")
        continue

    print(f"{name}: searching...")
    sr = lk.search_lightcurve(search_term, mission="TESS", author="SPOC")

    if len(sr) == 0:
        sr = lk.search_lightcurve(search_term, mission="TESS")

    if len(sr) == 0:
        print(f"{name}: no data found")
        continue

    print(f"{name}: {len(sr)} sectors found, downloading...")

    if stitch:
        lc = sr.download_all().stitch().remove_nans().remove_outliers(sigma=4)
    else:
        lc = sr[0].download().remove_nans().remove_outliers(sigma=4)

    lc_flat = lc.flatten(window_length=401)

    np.savez(out_path,
        time=lc.time.value,
        flux=lc.flux.value,
        flux_err=lc.flux_err.value,
        flat_flux=lc_flat.flux.value,
        period=period,
        name=name,
    )

    print(f"{name}: saved to {out_path}")

print("Done.")