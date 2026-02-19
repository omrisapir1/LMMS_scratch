from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import LMMSVocab


@dataclass
class Pass1Output:
    logits_all: torch.Tensor
    probs_all: torch.Tensor
    p_stop_unmasked: torch.Tensor
    action_probs: torch.Tensor
    action_ids: torch.Tensor
    alive_mask: torch.Tensor
    first_stop: torch.Tensor


class LMMSWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, vocab: LMMSVocab, kmax: int, vz: int):
        super().__init__()
        self.base_model = base_model
        self.vocab = vocab
        self.kmax = kmax
        self.vz = vz

        if len(vocab.z_token_ids) != vz:
            raise ValueError(f"Expected {vz} z tokens, got {len(vocab.z_token_ids)}")

        hidden_size = self._hidden_size()
        self.digit_heads = nn.ModuleList([nn.Linear(hidden_size, 10) for _ in range(5)])

        action_token_ids = torch.tensor(vocab.action_token_ids, dtype=torch.long)
        self.register_buffer("action_token_ids", action_token_ids, persistent=False)
        self.stop_action_index = vz

    def _hidden_size(self) -> int:
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(self.base_model.config, attr):
                return int(getattr(self.base_model.config, attr))
        raise ValueError("Could not infer hidden size from model config")

    def _lm_head_weight(self) -> torch.Tensor:
        out_emb = self.base_model.get_output_embeddings()
        if out_emb is None or not hasattr(out_emb, "weight"):
            raise ValueError("Model does not expose output embedding weights")
        return out_emb.weight

    def _gather_positions(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, _, hsz = hidden.shape
        idx = positions.unsqueeze(-1).expand(bsz, positions.size(1), hsz)
        return hidden.gather(dim=1, index=idx)

    def _gather_single_positions(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, _, hsz = hidden.shape
        idx = positions.view(bsz, 1, 1).expand(bsz, 1, hsz)
        return hidden.gather(dim=1, index=idx).squeeze(1)

    def forward_hidden(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Call the decoder backbone directly to get last hidden states
        # without materializing all intermediate hidden states.
        out = self.base_model.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    def latent_logits_from_hidden(self, latent_hidden: torch.Tensor) -> torch.Tensor:
        lm_w = self._lm_head_weight()
        rows = lm_w[self.action_token_ids]
        return torch.einsum("bkh,vh->bkv", latent_hidden, rows)

    def _force_stop(self, hard_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        hard = hard_actions.clone()
        stop_mask = hard[..., self.stop_action_index] > 0.5
        has_stop = stop_mask.any(dim=1)

        if (~has_stop).any():
            missing = (~has_stop).nonzero(as_tuple=False).squeeze(-1)
            hard[missing, -1, :] = 0.0
            hard[missing, -1, self.stop_action_index] = 1.0

        stop_mask = hard[..., self.stop_action_index] > 0.5
        first_stop = stop_mask.long().argmax(dim=1)
        t = torch.arange(self.kmax, device=hard.device).unsqueeze(0)
        alive_mask = t <= first_stop.unsqueeze(1)
        action_ids = hard.argmax(dim=-1)

        return {
            "hard_actions": hard,
            "action_ids": action_ids,
            "alive_mask": alive_mask,
            "first_stop": first_stop,
        }

    def sample_actions_gs_st(self, logits_all: torch.Tensor, tau: float) -> Dict[str, torch.Tensor]:
        g = -torch.empty_like(logits_all).exponential_().log()
        y_soft = F.softmax((logits_all + g) / tau, dim=-1)
        y_idx = y_soft.argmax(dim=-1)
        y_hard = F.one_hot(y_idx, num_classes=logits_all.size(-1)).to(logits_all.dtype)

        forced = self._force_stop(y_hard)
        y_st = forced["hard_actions"] + y_soft - y_soft.detach()

        return {
            "action_probs": y_st,
            "action_ids": forced["action_ids"],
            "alive_mask": forced["alive_mask"],
            "first_stop": forced["first_stop"],
        }

    def argmax_actions(self, logits_all: torch.Tensor) -> Dict[str, torch.Tensor]:
        idx = logits_all.argmax(dim=-1)
        y_hard = F.one_hot(idx, num_classes=logits_all.size(-1)).to(logits_all.dtype)
        forced = self._force_stop(y_hard)
        return {
            "action_probs": forced["hard_actions"],
            "action_ids": forced["action_ids"],
            "alive_mask": forced["alive_mask"],
            "first_stop": forced["first_stop"],
        }

    def pass1_policy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_positions: torch.Tensor,
        tau: float,
        deterministic: bool,
    ) -> Pass1Output:
        hidden = self.forward_hidden(
            input_ids=input_ids,
            inputs_embeds=None,
            attention_mask=attention_mask,
        )
        latent_hidden = self._gather_positions(hidden, latent_positions)
        logits_all = self.latent_logits_from_hidden(latent_hidden)
        probs_all = F.softmax(logits_all, dim=-1)
        p_stop_unmasked = probs_all[..., self.stop_action_index]

        if deterministic:
            act = self.argmax_actions(logits_all)
        else:
            act = self.sample_actions_gs_st(logits_all, tau=tau)

        return Pass1Output(
            logits_all=logits_all,
            probs_all=probs_all,
            p_stop_unmasked=p_stop_unmasked,
            action_probs=act["action_probs"],
            action_ids=act["action_ids"],
            alive_mask=act["alive_mask"],
            first_stop=act["first_stop"],
        )

    def inject_actions(
        self,
        input_ids: torch.Tensor,
        latent_positions: torch.Tensor,
        action_probs: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        in_emb = self.base_model.get_input_embeddings()
        full_emb = in_emb(input_ids).clone()

        action_emb_table = in_emb.weight[self.action_token_ids]
        latent_emb = torch.einsum("bkv,vh->bkh", action_probs, action_emb_table)
        latent_emb = latent_emb * alive_mask.unsqueeze(-1).to(latent_emb.dtype)

        bsz, kmax = latent_positions.shape
        batch_idx = torch.arange(bsz, device=latent_positions.device).unsqueeze(1).expand(bsz, kmax)
        full_emb[batch_idx, latent_positions] = latent_emb
        return full_emb

    def mask_dead_latent_attention(
        self,
        attention_mask: torch.Tensor,
        latent_positions: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_attention = attention_mask.clone()
        alive_bool = alive_mask.bool()
        dead = ~alive_bool
        if dead.any():
            bsz, kmax = latent_positions.shape
            batch_idx = torch.arange(
                bsz,
                device=latent_positions.device,
            ).unsqueeze(1).expand(bsz, kmax)
            masked_attention[batch_idx[dead], latent_positions[dead]] = 0
        return masked_attention

    def pass_with_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_positions: torch.Tensor,
        action_probs: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.inject_actions(
            input_ids=input_ids,
            latent_positions=latent_positions,
            action_probs=action_probs,
            alive_mask=alive_mask,
        )
        masked_attention = self.mask_dead_latent_attention(
            attention_mask=attention_mask,
            latent_positions=latent_positions,
            alive_mask=alive_mask,
        )
        return self.forward_hidden(
            input_ids=None,
            attention_mask=masked_attention,
            inputs_embeds=inputs_embeds,
        )

    def digit_logits_from_hidden(self, answer_hidden: torch.Tensor) -> torch.Tensor:
        logits = [head(answer_hidden) for head in self.digit_heads]
        return torch.stack(logits, dim=1)

    def extract_answer_hidden(self, hidden: torch.Tensor, answer_positions: torch.Tensor) -> torch.Tensor:
        return self._gather_single_positions(hidden, answer_positions)

    def hard_one_hot_from_ids(self, action_ids: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action_ids, num_classes=self.vz + 1).to(self._lm_head_weight().dtype)
