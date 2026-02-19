from __future__ import annotations

import random
from typing import Tuple

import torch


MODES = ("random_replacement", "permutation", "truncation")


def _first_stop_from_alive(alive_mask: torch.Tensor) -> torch.Tensor:
    # alive_mask: [B, K] with at least one alive slot
    return alive_mask.long().sum(dim=1) - 1


def perturb_actions(
    action_ids: torch.Tensor,
    alive_mask: torch.Tensor,
    stop_idx: int,
    vz: int,
    mode: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if mode is None:
        mode = random.choice(MODES)
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    bsz, kmax = action_ids.shape
    out = action_ids.clone()
    first_stop = _first_stop_from_alive(alive_mask)

    for b in range(bsz):
        s = int(first_stop[b].item())
        if s < 0:
            continue

        if mode == "random_replacement":
            if s > 0:
                out[b, :s] = torch.randint(0, vz, (s,), device=out.device)
            out[b, s] = stop_idx

        elif mode == "permutation":
            if s > 1:
                perm = torch.randperm(s, device=out.device)
                out[b, :s] = out[b, :s][perm]
            out[b, s] = stop_idx

        elif mode == "truncation":
            new_stop = s // 2
            out[b, new_stop] = stop_idx
            if new_stop + 1 < kmax:
                # Dead slots are masked by alive_cf during injection; this fill value is unused.
                out[b, new_stop + 1 :] = 0

    stop_mask = out == stop_idx
    has_stop = stop_mask.any(dim=1)
    if (~has_stop).any():
        missing = (~has_stop).nonzero(as_tuple=False).squeeze(-1)
        out[missing, -1] = stop_idx

    stop_mask = out == stop_idx
    first_stop_cf = stop_mask.long().argmax(dim=1)
    t = torch.arange(kmax, device=out.device).unsqueeze(0)
    alive_cf = t <= first_stop_cf.unsqueeze(1)
    return out, alive_cf, mode


def perturb_actions_deterministic(
    action_ids: torch.Tensor,
    alive_mask: torch.Tensor,
    stop_idx: int,
    vz: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    # Deterministic eval path: fixed truncation mode.
    del vz
    out = action_ids.clone()
    bsz, kmax = out.shape
    first_stop = _first_stop_from_alive(alive_mask)

    for b in range(bsz):
        s = int(first_stop[b].item())
        new_stop = s // 2
        out[b, new_stop] = stop_idx
        if new_stop + 1 < kmax:
            # Dead slots are masked by alive_cf during injection; this fill value is unused.
            out[b, new_stop + 1 :] = 0

    stop_mask = out == stop_idx
    has_stop = stop_mask.any(dim=1)
    if (~has_stop).any():
        missing = (~has_stop).nonzero(as_tuple=False).squeeze(-1)
        out[missing, -1] = stop_idx

    stop_mask = out == stop_idx
    first_stop_cf = stop_mask.long().argmax(dim=1)
    t = torch.arange(kmax, device=out.device).unsqueeze(0)
    alive_cf = t <= first_stop_cf.unsqueeze(1)
    return out, alive_cf, "truncation"
