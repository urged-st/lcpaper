import numpy as np
import glob

for path in sorted(glob.glob("lightcurves/*.npz")):
    d = np.load(path, allow_pickle=True)
    name = str(d["name"])
    time = d["time"]
    print(f"{name}: time range {time.min():.3f} to {time.max():.3f}, first={time[0]:.3f}")