from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from rl.ppo import sample_portfolios


def collect_train_rollout(policy, cache: List[dict], device, cfg: dict) -> List[dict]:
    """Rollout only on Train cache items. Test never enters this buffer."""
    g = int(cfg["rl"]["group_size"])
    std = float(cfg["rl"]["noise_std"])
    rows = []
    for item in cache:
        cond = torch.from_numpy(item["cond"]).to(device)
        w, logits, logp = sample_portfolios(policy, cond, g, std)
        rows.append(
            {
                "asof": item["asof"],
                "weights": w.detach(),
                "logits": logits.detach(),
                "logp": logp.detach(),
                "R": item["R"],
                "cond": item["cond"],
                "split": "train",
            }
        )
    return rows
