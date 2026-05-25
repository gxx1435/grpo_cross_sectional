"""Time each stage of the data pipeline so we can see where the cost is."""

import sys, time
from pathlib import Path

sys.path.insert(0, "files")

t0 = time.time()
print(f"[{time.time()-t0:6.2f}s] start", flush=True)

import numpy as np
import pandas as pd
import torch
print(f"[{time.time()-t0:6.2f}s] numpy/pandas/torch imported", flush=True)

from hf_data import HFConfig, build_hf_dataset
print(f"[{time.time()-t0:6.2f}s] hf_data imported", flush=True)

cfg = HFConfig(
    pool_dir=Path("files/csi500/2026_1min"),
    start_day=0,
    num_days=10,
    n_stocks_max=30,
    lookback=30,
    horizon=5,
)
print(f"[{time.time()-t0:6.2f}s] cfg built; building dataset...", flush=True)

bundle = build_hf_dataset(cfg)
print(f"[{time.time()-t0:6.2f}s] dataset built. shapes:", flush=True)
for k, v in bundle.items():
    if isinstance(v, np.ndarray):
        print(f"    {k:14s} shape={tuple(v.shape)} dtype={v.dtype}", flush=True)
print(f"[{time.time()-t0:6.2f}s] done", flush=True)
