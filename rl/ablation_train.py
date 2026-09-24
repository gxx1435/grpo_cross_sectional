"""SP500 RL ablation trainers: candidate-feature PPO, select PPO, SS-FM-init GRPO."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backtest.drawdown import smooth_dd_to_date
from backtest.execution import execute_day
from backtest.metrics import causal_sharpe
from experiments.engine import cond_dim, select_candidate
from experiments.progress import Progress
from flow_matching.ssfm import sample_ss_fm_mixed
from portfolio.teacher_portfolios import TEACHER_NAMES
from rl.candidate_policy import CandidateWeightPolicy, ppo_update_candidate, sample_portfolios_from_state
from rl.grpo import grpo_update
from rl.ppo import ValueHead, WeightPolicy, sample_portfolios
from rl.reward import composite_reward, fit_reward_scalers, reward_mode
from rl.select_policy import CandidateSelectPolicy, ppo_update_select, sample_indices
from utils.logging import log


def _reward_scalers_warmup(cache: List[dict], k: int, cfg: dict):
    hist = {"return": [], "sharpe": [], "turnover": [], "smooth_mdd": []}
    prev = None
    nets = []
    for item in cache:
        w = np.ones(k) / k
        met = execute_day(w, item["R"], prev, cfg)
        nets.append(met["net_return"])
        hist["return"].append(met["net_return"])
        hist["sharpe"].append(
            causal_sharpe(
                nets[:-1],
                cfg["portfolio"]["sharpe_min_obs"],
                0.0,
                int(cfg["portfolio"].get("sharpe_window", 20)),
            )
        )
        hist["turnover"].append(met["turnover"])
        hist["smooth_mdd"].append(smooth_dd_to_date(np.asarray(nets[:-1]), cfg["portfolio"]["smooth_dd_temperature"]))
        prev = w
    fit_range = {"start": str(cache[0]["asof"].date()), "end": str(cache[-1]["asof"].date())}
    return fit_reward_scalers({kk: np.asarray(v) for kk, v in hist.items()}, cfg, fit_range)


def _group_rewards(ws, item, prev, hist_net, scalers, cfg, g: int):
    rews = []
    mode = reward_mode(cfg)
    shp_shared = causal_sharpe(
        hist_net,
        cfg["portfolio"]["sharpe_min_obs"],
        0.0,
        int(cfg["portfolio"].get("sharpe_window", 20)),
    )
    smdd = smooth_dd_to_date(np.asarray(hist_net), cfg["portfolio"]["smooth_dd_temperature"])
    for gi in range(g):
        met = execute_day(ws[gi], item["R"], prev, cfg)
        if mode in ("sharpe", "sharpe_only", "single_sharpe"):
            shp = causal_sharpe(
                list(hist_net) + [met["net_return"]],
                cfg["portfolio"]["sharpe_min_obs"],
                0.0,
                int(cfg["portfolio"].get("sharpe_window", 20)),
            )
        else:
            shp = shp_shared
        pack = composite_reward(
            {"return": met["net_return"], "sharpe": shp, "turnover": met["turnover"], "smooth_mdd": smdd},
            scalers,
            cfg,
        )
        rews.append(pack["reward"])
    return rews


def _pick_prev(ws, rews, item, cfg, prev_mode: str):
    if prev_mode == "best_reward":
        j = int(np.argmax(np.asarray(rews, dtype=np.float64)))
        w_next = np.asarray(ws[j], dtype=np.float64)
    elif prev_mode == "select_candidate" and "alpha" in item and "idx" in item and "sigma" in item:
        _, w_next = select_candidate(ws, item["alpha"][item["idx"]], item["sigma"], cfg)
        w_next = np.asarray(w_next, dtype=np.float64)
    else:
        w_next = np.asarray(ws.mean(0), dtype=np.float64)
    w_next = np.clip(w_next, 0, None)
    return w_next / max(float(w_next.sum()), 1e-12)


def _ssfm_candidates(ss_fm, cond, g: int, n_steps: int, item, cfg, device):
    with torch.no_grad():
        ws_t = sample_ss_fm_mixed(ss_fm, cond, g, n_steps).detach()
    arr = ws_t.cpu().numpy()
    if "alpha" in item and "idx" in item and "sigma" in item:
        _, w_sel = select_candidate(arr, item["alpha"][item["idx"]], item["sigma"], cfg)
    else:
        w_sel = arr[0]
    return ws_t, np.asarray(w_sel, dtype=np.float32)


def _meta_candidate_pool(
    ss_fm,
    cond,
    g: int,
    n_steps: int,
    item,
    device,
    include_teachers: bool = True,
) -> torch.Tensor:
    """Build Meta-style pool: G SS-FM samples ∪ teacher closed-forms. Shape (N, K)."""
    with torch.no_grad():
        ws_t = sample_ss_fm_mixed(ss_fm, cond, g, n_steps).detach()
    parts = [ws_t]
    if include_teachers:
        teachers = item.get("teachers") or {}
        for tn in TEACHER_NAMES:
            if tn not in teachers:
                continue
            w = np.asarray(teachers[tn], dtype=np.float32).reshape(-1)
            if w.size == ws_t.shape[-1]:
                parts.append(torch.from_numpy(w).to(device=device, dtype=ws_t.dtype).unsqueeze(0))
    return torch.cat(parts, dim=0) if len(parts) > 1 else ws_t


def _meta_argmax(pool: torch.Tensor, item, cfg) -> int:
    arr = pool.detach().cpu().numpy()
    j, _ = select_candidate(arr, item["alpha"][item["idx"]], item["sigma"], cfg)
    return int(j)


def train_ppo_candidate(
    ss_fm,
    cache: List[dict],
    cfg: dict,
    device,
    k: Optional[int] = None,
    compact: bool = False,
) -> CandidateWeightPolicy:
    """PPO: frozen SS-FM candidates as features → policy re-outputs weights."""
    k = int(k if k is not None else cfg["portfolio"]["top_k"])
    cd = cond_dim(k, compact=compact)
    hidden = int(cfg["rl"]["hidden"])
    pol = CandidateWeightPolicy(cd, k, hidden).to(device)
    critic = ValueHead(pol.state_dim, hidden).to(device)
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    g_cand = int(cfg["rl"].get("candidate_group_size") or cfg["rl"]["group_size"])
    g = int(cfg["rl"]["group_size"])
    opt_d = torch.optim.Adam(pol.parameters(), lr=1e-3)
    dist_bar = Progress(int(cfg["rl"]["distill_steps"]), "D-RL ppo_cand distill", log_every=20)
    for di in range(int(cfg["rl"]["distill_steps"])):
        item = cache[np.random.randint(0, len(cache))]
        cond = torch.from_numpy(item["cond"]).to(device)
        cands, w_np = _ssfm_candidates(ss_fm, cond, g_cand, n_steps, item, cfg, device)
        state = pol.encode(cond, cands)
        w = torch.from_numpy(w_np).to(device)
        tgt = torch.log(w.clamp_min(1e-8))
        tgt = tgt - tgt.mean()
        loss = F.mse_loss(pol.forward_logits(state).squeeze(0), tgt)
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
        dist_bar.update(di + 1, f"loss={float(loss):.4f}")
    dist_bar.close()

    opt = torch.optim.Adam(list(pol.parameters()) + list(critic.parameters()), lr=float(cfg["rl"]["lr"]))
    scalers = _reward_scalers_warmup(cache, k, cfg)
    prev_mode = str(cfg["rl"].get("prev_mode", "mean") or "mean").lower()
    rl_bar = Progress(int(cfg["rl"]["epochs"]) * max(len(cache), 1), "D-RL PPO_CAND", log_every=10)
    for ep in range(int(cfg["rl"]["epochs"])):
        n_upd = 0
        t0 = time.time()
        prev = None
        hist_net = []
        for item in cache:
            cond = torch.from_numpy(item["cond"]).to(device)
            cands, _ = _ssfm_candidates(ss_fm, cond, g_cand, n_steps, item, cfg, device)
            with torch.no_grad():
                state0 = pol.encode(cond, cands)
            w, logits_s, _ = sample_portfolios_from_state(pol, state0, g, float(cfg["rl"]["noise_std"]))
            ws = w.detach().cpu().numpy()
            rews = _group_rewards(ws, item, prev, hist_net, scalers, cfg, g)
            rew_t = torch.tensor(rews, device=device, dtype=torch.float32)
            ppo_update_candidate(pol, critic, cond, cands, logits_s, rew_t, opt, cfg)
            w_next = _pick_prev(ws, rews, item, cfg, prev_mode)
            met0 = execute_day(w_next, item["R"], prev, cfg)
            hist_net.append(met0["net_return"])
            prev = w_next
            n_upd += 1
            rl_bar.update(ep * len(cache) + n_upd, f"ep{ep+1} {item['asof'].date()}")
        log(f"      PPO_CAND epoch {ep+1} updates={n_upd} wall={time.time()-t0:.1f}s mode={reward_mode(cfg)}")
    rl_bar.close()
    pol._reward_scalers = scalers  # type: ignore[attr-defined]
    return pol


def train_ppo_select(
    ss_fm,
    cache: List[dict],
    cfg: dict,
    device,
    k: Optional[int] = None,
    compact: bool = False,
    include_teachers: bool = True,
) -> CandidateSelectPolicy:
    """PPO discrete selector over Meta pool (SS-FM samples ∪ teachers).

    Does NOT emit new weights — only picks an index into the frozen candidate set.
    Distill target = Meta pred_utility argmax.
    """
    k = int(k if k is not None else cfg["portfolio"]["top_k"])
    cd = cond_dim(k, compact=compact)
    hidden = int(cfg["rl"]["hidden"])
    pol = CandidateSelectPolicy(cd, k, hidden).to(device)
    critic = ValueHead(pol.state_dim, hidden).to(device)
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    g_cand = int(cfg["rl"].get("candidate_group_size") or cfg["rl"]["group_size"])
    g = int(cfg["rl"]["group_size"])

    opt_d = torch.optim.Adam(pol.parameters(), lr=1e-3)
    dist_bar = Progress(int(cfg["rl"]["distill_steps"]), "D-RL ppo_select distill", log_every=20)
    for di in range(int(cfg["rl"]["distill_steps"])):
        item = cache[np.random.randint(0, len(cache))]
        cond = torch.from_numpy(item["cond"]).to(device)
        pool = _meta_candidate_pool(ss_fm, cond, g_cand, n_steps, item, device, include_teachers=include_teachers)
        tgt_j = _meta_argmax(pool, item, cfg)
        scores = pol.forward_scores(cond, pool)
        loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([tgt_j], device=device, dtype=torch.long))
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
        dist_bar.update(di + 1, f"loss={float(loss):.4f}")
    dist_bar.close()

    opt = torch.optim.Adam(list(pol.parameters()) + list(critic.parameters()), lr=float(cfg["rl"]["lr"]))
    scalers = _reward_scalers_warmup(cache, k, cfg)
    prev_mode = str(cfg["rl"].get("prev_mode", "mean") or "mean").lower()
    rl_bar = Progress(int(cfg["rl"]["epochs"]) * max(len(cache), 1), "D-RL PPO_SELECT", log_every=10)
    for ep in range(int(cfg["rl"]["epochs"])):
        n_upd = 0
        t0 = time.time()
        prev = None
        hist_net = []
        for item in cache:
            cond = torch.from_numpy(item["cond"]).to(device)
            pool = _meta_candidate_pool(ss_fm, cond, g_cand, n_steps, item, device, include_teachers=include_teachers)
            idxs, _, _ = sample_indices(pol, cond, pool, g)
            ws = pool[idxs].detach().cpu().numpy()
            rews = _group_rewards(ws, item, prev, hist_net, scalers, cfg, g)
            rew_t = torch.tensor(rews, device=device, dtype=torch.float32)
            ppo_update_select(pol, critic, cond, pool, idxs, rew_t, opt, cfg)
            if prev_mode == "best_reward":
                j = int(np.argmax(np.asarray(rews, dtype=np.float64)))
                w_next = np.asarray(ws[j], dtype=np.float64)
            elif prev_mode == "select_candidate":
                # use Meta argmax on the same pool for prev (stable teacher)
                w_next = pool[_meta_argmax(pool, item, cfg)].detach().cpu().numpy().astype(np.float64)
            else:
                # policy argmax
                with torch.no_grad():
                    _, w_t = pol.select(cond, pool)
                w_next = w_t.detach().cpu().numpy().astype(np.float64)
            w_next = np.clip(w_next, 0, None)
            w_next = w_next / max(float(w_next.sum()), 1e-12)
            met0 = execute_day(w_next, item["R"], prev, cfg)
            hist_net.append(met0["net_return"])
            prev = w_next
            n_upd += 1
            rl_bar.update(ep * len(cache) + n_upd, f"ep{ep+1} {item['asof'].date()}")
        log(f"      PPO_SELECT epoch {ep+1} updates={n_upd} wall={time.time()-t0:.1f}s mode={reward_mode(cfg)}")
    rl_bar.close()
    pol._reward_scalers = scalers  # type: ignore[attr-defined]
    return pol


def train_grpo_ssfm_init(
    ss_fm,
    cache: List[dict],
    cfg: dict,
    device,
    k: Optional[int] = None,
    compact: bool = False,
) -> WeightPolicy:
    """GRPO with WeightPolicy distilled from SS-FM as the initial policy (no online BC)."""
    k = int(k if k is not None else cfg["portfolio"]["top_k"])
    cd = cond_dim(k, compact=compact)
    pol = WeightPolicy(cd, k, int(cfg["rl"]["hidden"])).to(device)
    opt_d = torch.optim.Adam(pol.parameters(), lr=1e-3)
    n_bc = max(int(cfg["rl"].get("teacher_bc_samples", 4)), 1)
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    dist_bar = Progress(int(cfg["rl"]["distill_steps"]), "D-RL grpo_init distill", log_every=20)
    for di in range(int(cfg["rl"]["distill_steps"])):
        item = cache[np.random.randint(0, len(cache))]
        cond = torch.from_numpy(item["cond"]).to(device)
        with torch.no_grad():
            ws_t = sample_ss_fm_mixed(ss_fm, cond, n_bc, n_steps).detach().cpu().numpy()
            if "alpha" in item and "idx" in item and "sigma" in item:
                _, w_np = select_candidate(ws_t, item["alpha"][item["idx"]], item["sigma"], cfg)
            else:
                w_np = ws_t[0]
            w = torch.from_numpy(np.asarray(w_np, dtype=np.float32)).to(device)
        tgt = torch.log(w.clamp_min(1e-8))
        tgt = tgt - tgt.mean()
        loss = F.mse_loss(pol.forward_logits(cond).squeeze(0), tgt)
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
        dist_bar.update(di + 1, f"loss={float(loss):.4f}")
    dist_bar.close()

    opt = torch.optim.Adam(pol.parameters(), lr=float(cfg["rl"]["lr"]))
    scalers = _reward_scalers_warmup(cache, k, cfg)
    g = int(cfg["rl"]["group_size"])
    prev_mode = str(cfg["rl"].get("prev_mode", "mean") or "mean").lower()
    rl_bar = Progress(int(cfg["rl"]["epochs"]) * max(len(cache), 1), "D-RL GRPO_INIT", log_every=10)
    for ep in range(int(cfg["rl"]["epochs"])):
        n_upd = 0
        t0 = time.time()
        prev = None
        hist_net = []
        for item in cache:
            cond = torch.from_numpy(item["cond"]).to(device)
            w, logits_s, _ = sample_portfolios(pol, cond, g, float(cfg["rl"]["noise_std"]))
            ws = w.detach().cpu().numpy()
            rews = _group_rewards(ws, item, prev, hist_net, scalers, cfg, g)
            rew_t = torch.tensor(rews, device=device, dtype=torch.float32)
            info = grpo_update(pol, cond, logits_s, rew_t, opt, cfg)
            if info.get("warning"):
                log(f"      GRPO_INIT group-relative signal weak asof={item['asof'].date()}")
            w_next = _pick_prev(ws, rews, item, cfg, prev_mode)
            met0 = execute_day(w_next, item["R"], prev, cfg)
            hist_net.append(met0["net_return"])
            prev = w_next
            n_upd += 1
            rl_bar.update(ep * len(cache) + n_upd, f"ep{ep+1} {item['asof'].date()}")
        log(f"      GRPO_INIT epoch {ep+1} updates={n_upd} wall={time.time()-t0:.1f}s mode={reward_mode(cfg)}")
    rl_bar.close()
    pol._reward_scalers = scalers  # type: ignore[attr-defined]
    return pol
