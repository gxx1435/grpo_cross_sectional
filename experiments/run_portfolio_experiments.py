from __future__ import annotations

from pathlib import Path

from data.splits import month_starts
from data.store import ResearchStore
from experiments.engine import run_month
from experiments.progress import Progress
from utils.logging import log


def run_portfolio_experiments(cfg: dict, store: ResearchStore, out_root: Path, oos_mode: str, months=None) -> None:
    months = months or month_starts(cfg["walkforward"]["test_start"], cfg["walkforward"]["test_end"])
    prog = Progress(len(months), f"B-port {oos_mode}", log_every=1)
    for m in months:
        log(f"== portfolio baselines {oos_mode} {m.strftime('%Y-%m')}")
        run_month(cfg, store, m, oos_mode, out_root, models_filter=[cfg["prediction"]["primary_model"]], skip_rl=True, skip_gen=True)
        prog.update(msg=m.strftime("%Y-%m"))
    prog.close()
