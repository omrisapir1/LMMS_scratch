from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def answer_loss(
    digit_logits: torch.Tensor,
    answer_digits: torch.Tensor,
    keep_prob: tuple[float, ...] | None = None,
) -> torch.Tensor:
    # digit_logits: [B, 5, 10], answer_digits: [B, 5]
    per_digit = []
    for d in range(5):
        per_digit.append(F.cross_entropy(digit_logits[:, d, :], answer_digits[:, d]))
    per_digit_loss = torch.stack(per_digit)

    if keep_prob is None:
        return per_digit_loss.mean()

    if len(keep_prob) != 5:
        raise ValueError(f"keep_prob must contain 5 values, got {len(keep_prob)}")

    keep = torch.tensor(keep_prob, device=digit_logits.device, dtype=per_digit_loss.dtype)
    if (keep < 0).any() or (keep > 1).any():
        raise ValueError("keep_prob values must be in [0,1]")

    mask = torch.bernoulli(keep)
    # Avoid all-dropped edge case: keep at least one digit loss active.
    if mask.sum().item() == 0:
        mask[-1] = 1.0

    weighted = per_digit_loss * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def js_divergence_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)

    eps = 1e-8
    p_log = torch.log(p + eps)
    q_log = torch.log(q + eps)
    m_log = torch.log(m + eps)

    kl_pm = (p * (p_log - m_log)).sum(dim=-1)
    kl_qm = (q * (q_log - m_log)).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return js.mean()


def counterfactual_loss(ref_digit_logits: torch.Tensor, cf_digit_logits: torch.Tensor) -> torch.Tensor:
    # Freeze the reference branch for the JS term.
    js = js_divergence_from_logits(ref_digit_logits.detach(), cf_digit_logits)
    return math.log(2.0) - js


def compute_loss_unmasked_stop(
    p_stop_unmasked: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    # p_stop_unmasked: [B, K]
    bsz, kmax = p_stop_unmasked.shape
    survival = torch.ones_like(p_stop_unmasked)
    if kmax > 1:
        survival[:, 1:] = torch.cumprod(1.0 - p_stop_unmasked[:, :-1], dim=1)

    # Power-law weighting in step index (1-based to keep the first term non-zero).
    steps = torch.arange(1, kmax + 1, device=p_stop_unmasked.device, dtype=p_stop_unmasked.dtype)
    weights = steps.pow(gamma)
    weighted_sum = (survival * weights.unsqueeze(0)).sum(dim=1)

    # Normalize by maximum possible sum of weights.
    max_sum = weights.sum().clamp_min(1e-8)
    normalized = weighted_sum / max_sum

    return normalized.mean()


def batch_collision_loss(
    action_ids: torch.Tensor,
    alive_mask: torch.Tensor,
    stop_action_index: int,
    vz: int,
) -> torch.Tensor:
    # Herfindahl on deterministic hard actions over alive Z slots only.
    out_dtype = torch.bfloat16
    valid = alive_mask & (action_ids != stop_action_index)
    if not valid.any():
        return torch.zeros((), device=action_ids.device, dtype=out_dtype)

    z_actions = action_ids[valid]
    z_actions = z_actions[(z_actions >= 0) & (z_actions < vz)]
    if z_actions.numel() == 0:
        return torch.zeros((), device=action_ids.device, dtype=out_dtype)

    counts = torch.bincount(z_actions, minlength=vz).to(out_dtype)
    p_v = counts / counts.sum().clamp_min(1.0)
    return (p_v.pow(2)).sum()


def digit_accuracy(digit_logits: torch.Tensor, answer_digits: torch.Tensor) -> torch.Tensor:
    pred = digit_logits.argmax(dim=-1)
    exact = (pred == answer_digits).all(dim=1)
    return exact.to(digit_logits.dtype).mean()


def combine_total_loss(
    loss_answer: torch.Tensor,
    loss_cf: torch.Tensor,
    loss_compute: torch.Tensor,
    loss_batch: torch.Tensor,
    w_answer: float,
    w_cf: float,
    w_compute: float,
    w_batch: float,
) -> torch.Tensor:
    return (
        w_answer * loss_answer
        + w_cf * loss_cf
        + w_compute * loss_compute
        + w_batch * loss_batch
    )


def loss_dict(
    total: torch.Tensor,
    answer: torch.Tensor,
    cf: torch.Tensor,
    compute: torch.Tensor,
    batch: torch.Tensor,
    accuracy: torch.Tensor,
    stop_mean: torch.Tensor,
) -> Dict[str, float]:
    return {
        "total": float(total.detach().cpu()),
        "answer": float(answer.detach().cpu()),
        "cf": float(cf.detach().cpu()),
        "compute": float(compute.detach().cpu()),
        "batch": float(batch.detach().cpu()),
        "accuracy": float(accuracy.detach().cpu()),
        "stop_mean": float(stop_mean.detach().cpu()),
    }
