from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterator, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LMMSConfig
from .counterfactual import perturb_actions, perturb_actions_deterministic
from .data import LMMSDataset, LMMScollator
from .losses import (
    answer_loss,
    batch_collision_loss,
    combine_total_loss,
    compute_loss_unmasked_stop,
    counterfactual_loss,
    digit_accuracy,
    loss_dict,
)
from .model import LMMSWrapper
from .tokenizer import extend_tokenizer, format_prompt


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _iter_forever(loader: DataLoader) -> Iterator[Dict]:
    while True:
        for batch in loader:
            yield batch


@contextmanager
def _temporary_use_cache(base_model, enabled: bool):
    prev = getattr(base_model.config, "use_cache", None)
    if prev is not None:
        base_model.config.use_cache = enabled
    try:
        yield
    finally:
        if prev is not None:
            base_model.config.use_cache = prev


def _cf_warmup_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step) / float(warmup_steps))


def run_three_pass(
    model: LMMSWrapper,
    batch: Dict,
    cfg: LMMSConfig,
    deterministic: bool,
    global_step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pass1 = model.pass1_policy(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        latent_positions=batch["latent_positions"],
        tau=cfg.model.tau,
        deterministic=deterministic,
    )

    # Pass 3: counterfactual injection
    if deterministic:
        cf_action_ids, cf_alive_mask, _ = perturb_actions_deterministic(
            action_ids=pass1.action_ids,
            alive_mask=pass1.alive_mask,
            stop_idx=model.stop_action_index,
            vz=cfg.model.Vz,
        )
    else:
        cf_action_ids, cf_alive_mask, _ = perturb_actions(
            action_ids=pass1.action_ids,
            alive_mask=pass1.alive_mask,
            stop_idx=model.stop_action_index,
            vz=cfg.model.Vz,
            mode=None,
        )

    # Pass 2+3 combined in one transformer forward over concatenated batch (2B).
    ref_inputs_embeds = model.inject_actions(
        input_ids=batch["input_ids"],
        latent_positions=batch["latent_positions"],
        action_probs=pass1.action_probs,
        alive_mask=pass1.alive_mask,
    )
    ref_attention_mask = model.mask_dead_latent_attention(
        attention_mask=batch["attention_mask"],
        latent_positions=batch["latent_positions"],
        alive_mask=pass1.alive_mask,
    )

    cf_action_probs = model.hard_one_hot_from_ids(cf_action_ids)
    cf_inputs_embeds = model.inject_actions(
        input_ids=batch["input_ids"],
        latent_positions=batch["latent_positions"],
        action_probs=cf_action_probs,
        alive_mask=cf_alive_mask,
    )
    cf_attention_mask = model.mask_dead_latent_attention(
        attention_mask=batch["attention_mask"],
        latent_positions=batch["latent_positions"],
        alive_mask=cf_alive_mask,
    )

    cat_inputs_embeds = torch.cat([ref_inputs_embeds, cf_inputs_embeds], dim=0)
    cat_attention_mask = torch.cat([ref_attention_mask, cf_attention_mask], dim=0)
    cat_hidden = model.forward_hidden(
        input_ids=None,
        attention_mask=cat_attention_mask,
        inputs_embeds=cat_inputs_embeds,
    )
    cat_answer_positions = torch.cat([batch["answer_positions"], batch["answer_positions"]], dim=0)
    cat_answer_hidden = model.extract_answer_hidden(cat_hidden, cat_answer_positions)
    cat_digit_logits = model.digit_logits_from_hidden(cat_answer_hidden)

    bsz = batch["input_ids"].size(0)
    ref_digit_logits = cat_digit_logits[:bsz]
    cf_digit_logits = cat_digit_logits[bsz:]

    l_answer = answer_loss(ref_digit_logits, batch["answer_digits"])
    l_cf = counterfactual_loss(ref_digit_logits, cf_digit_logits)
    l_compute = compute_loss_unmasked_stop(
        p_stop_unmasked=pass1.p_stop_unmasked,
        gamma=cfg.loss.gamma,
        lambda_compute=cfg.loss.lambda_compute,
    )
    batch_actions = model.argmax_actions(pass1.logits_all)
    l_batch = batch_collision_loss(
        action_ids=batch_actions["action_ids"],
        alive_mask=batch_actions["alive_mask"],
        stop_action_index=model.stop_action_index,
        vz=cfg.model.Vz,
    )

    if deterministic:
        cf_scale = 1.0
    else:
        cf_scale = _cf_warmup_scale(global_step, cfg.train.cf_warmup_steps)

    total = combine_total_loss(
        loss_answer=l_answer,
        loss_cf=l_cf * cf_scale,
        loss_compute=l_compute,
        loss_batch=l_batch,
        w_answer=cfg.loss.w_answer,
        w_cf=cfg.loss.w_cf,
        w_compute=cfg.loss.w_compute,
        w_batch=cfg.loss.w_batch,
    )

    acc = digit_accuracy(ref_digit_logits, batch["answer_digits"])
    stop_mean = (pass1.first_stop.to(pass1.logits_all.dtype) + 1.0).mean()
    metrics = loss_dict(
        total=total,
        answer=l_answer,
        cf=l_cf,
        compute=l_compute,
        batch=l_batch,
        accuracy=acc,
        stop_mean=stop_mean,
    )
    metrics["cf_scale"] = cf_scale
    return total, metrics


def evaluate(model: LMMSWrapper, loader: DataLoader, cfg: LMMSConfig, device: torch.device) -> Dict[str, float]:
    model.eval()
    # Explicitly enable KV cache on eval path.
    with _temporary_use_cache(model.base_model, True):
        sums: Dict[str, float] = {
            "total": 0.0,
            "answer": 0.0,
            "cf": 0.0,
            "compute": 0.0,
            "batch": 0.0,
            "accuracy": 0.0,
            "stop_mean": 0.0,
        }
        total_n = 0

        with torch.no_grad():
            for batch in loader:
                batch = _move_batch_to_device(batch, device)
                _, m = run_three_pass(
                    model=model,
                    batch=batch,
                    cfg=cfg,
                    deterministic=True,
                    global_step=0,
                )
                n = int(batch["input_ids"].size(0))
                total_n += n
                for k in sums:
                    sums[k] += m[k] * n

    if total_n == 0:
        return {k: 0.0 for k in sums}
    return {k: v / total_n for k, v in sums.items()}


def _get_first_answer_pos(ids: List[int], answer_token_id: int, prompt_len: int) -> int | None:
    for i in range(prompt_len, len(ids)):
        if ids[i] == answer_token_id:
            return i
    return None


def _digit_preds_from_batch(
    model: LMMSWrapper,
    token_ids_batch: List[List[int]],
    answer_pos_batch: List[int | None],
    pad_token_id: int,
    device: torch.device,
) -> List[List[int] | None]:
    out: List[List[int] | None] = [None] * len(token_ids_batch)
    valid = [i for i, pos in enumerate(answer_pos_batch) if pos is not None]
    if not valid:
        return out

    max_len = max(len(token_ids_batch[i]) for i in valid)
    inp = torch.full((len(valid), max_len), pad_token_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(valid), max_len), dtype=torch.long, device=device)
    answer_positions = []

    for row_idx, i in enumerate(valid):
        ids = token_ids_batch[i]
        seq_len = len(ids)
        inp[row_idx, :seq_len] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[row_idx, :seq_len] = 1
        answer_positions.append(int(answer_pos_batch[i]))

    with torch.no_grad():
        hidden = model.forward_hidden(input_ids=inp, attention_mask=attn)
        h_ans = model.extract_answer_hidden(
            hidden,
            torch.tensor(answer_positions, dtype=torch.long, device=device),
        )
        logits = model.digit_logits_from_hidden(h_ans)
    preds = logits.argmax(dim=-1).tolist()

    for row_idx, i in enumerate(valid):
        out[i] = preds[row_idx]
    return out


def _randomize_z_tokens(token_ids: List[int], z_token_ids: List[int], start: int, end: int) -> List[int]:
    z_set = set(z_token_ids)
    out = token_ids.copy()
    for i in range(start, end):
        if out[i] in z_set:
            out[i] = random.choice(z_token_ids)
    return out


def _truncate_z_tokens(token_ids: List[int], z_token_ids: List[int], start: int, end: int) -> List[int]:
    z_set = set(z_token_ids)
    keep = token_ids[:start]
    middle = [tok for tok in token_ids[start:end] if tok not in z_set]
    keep.extend(middle)
    keep.extend(token_ids[end:])
    return keep


def _generate_once(
    model: LMMSWrapper,
    tokenizer,
    vocab,
    prompt_ids: List[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[int]:
    eos_ids = [vocab.answer_token_id]
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(inp)
    gen_kwargs = {
        "input_ids": inp,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    # Explicitly enable KV cache for generation.
    with _temporary_use_cache(model.base_model, True):
        with torch.no_grad():
            generated = model.base_model.generate(**gen_kwargs)
    return generated[0].tolist()


def generate_eval_artifact(
    model: LMMSWrapper,
    tokenizer,
    vocab,
    eval_dataset: LMMSDataset,
    cfg: LMMSConfig,
    device: torch.device,
    global_step: int,
) -> str:
    model.eval()
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.train.output_dir, f"eval_artifact_step{global_step}_{ts}.jsonl")

    with _temporary_use_cache(model.base_model, True):
        n = min(cfg.train.artifact_eval_examples, len(eval_dataset))
        rows = []

        for i in range(n):
            row = eval_dataset.ds[i]
            question = row["question"]
            answer_digits = [int(x) for x in row["answer_digits"]]

            prompt_text = format_prompt(tokenizer, question)
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=cfg.data.max_length,
            )["input_ids"]

            greedy_ids = _generate_once(
                model=model,
                tokenizer=tokenizer,
                vocab=vocab,
                prompt_ids=prompt_ids,
                max_new_tokens=cfg.train.eval_generate_max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                device=device,
            )
            sample_ids = _generate_once(
                model=model,
                tokenizer=tokenizer,
                vocab=vocab,
                prompt_ids=prompt_ids,
                max_new_tokens=cfg.train.eval_generate_max_new_tokens,
                do_sample=True,
                temperature=cfg.train.eval_generate_temperature,
                top_p=cfg.train.eval_generate_top_p,
                device=device,
            )

            def build_variant(ids: List[int]) -> Dict[str, List[int] | int | None | str]:
                ans_pos = _get_first_answer_pos(ids, vocab.answer_token_id, len(prompt_ids))
                ans_end = ans_pos if ans_pos is not None else len(ids)

                rand_ids = _randomize_z_tokens(ids, vocab.z_token_ids, len(prompt_ids), ans_end)
                rand_ans = _get_first_answer_pos(rand_ids, vocab.answer_token_id, len(prompt_ids))

                trunc_ids = _truncate_z_tokens(ids, vocab.z_token_ids, len(prompt_ids), ans_end)
                trunc_ans = _get_first_answer_pos(trunc_ids, vocab.answer_token_id, len(prompt_ids))
                return {
                    "full_text": tokenizer.decode(ids, skip_special_tokens=False),
                    "digit_ids": ids,
                    "digit_ans": ans_pos,
                    "rand_text": tokenizer.decode(rand_ids, skip_special_tokens=False),
                    "rand_ids": rand_ids,
                    "rand_ans": rand_ans,
                    "trunc_text": tokenizer.decode(trunc_ids, skip_special_tokens=False),
                    "trunc_ids": trunc_ids,
                    "trunc_ans": trunc_ans,
                }

            greedy_v = build_variant(greedy_ids)
            sample_v = build_variant(sample_ids)

            digit_preds = _digit_preds_from_batch(
                model=model,
                token_ids_batch=[
                    greedy_v["digit_ids"],
                    sample_v["digit_ids"],
                    greedy_v["rand_ids"],
                    sample_v["rand_ids"],
                    greedy_v["trunc_ids"],
                    sample_v["trunc_ids"],
                ],
                answer_pos_batch=[
                    greedy_v["digit_ans"],
                    sample_v["digit_ans"],
                    greedy_v["rand_ans"],
                    sample_v["rand_ans"],
                    greedy_v["trunc_ans"],
                    sample_v["trunc_ans"],
                ],
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            (
                greedy_digit,
                sample_digit,
                greedy_rand_digit,
                sample_rand_digit,
                greedy_trunc_digit,
                sample_trunc_digit,
            ) = digit_preds

            rows.append(
                {
                    "question": question,
                    "answer_digits": answer_digits,
                    "greedy_full_text": greedy_v["full_text"],
                    "greedy_digit_pred": greedy_digit,
                    "sample_full_text": sample_v["full_text"],
                    "sample_digit_pred": sample_digit,
                    "greedy_randomized_full_text": greedy_v["rand_text"],
                    "greedy_randomized_digit_pred": greedy_rand_digit,
                    "sample_randomized_full_text": sample_v["rand_text"],
                    "sample_randomized_digit_pred": sample_rand_digit,
                    "greedy_truncated_full_text": greedy_v["trunc_text"],
                    "greedy_truncated_digit_pred": greedy_trunc_digit,
                    "sample_truncated_full_text": sample_v["trunc_text"],
                    "sample_truncated_digit_pred": sample_trunc_digit,
                }
            )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    return out_path


def _print_metrics(prefix: str, step: int, m: Dict[str, float]) -> None:
    print(
        (
            f"[{prefix}] step={step} "
            f"total={m['total']:.4f} answer={m['answer']:.4f} "
            f"cf={m['cf']:.4f} compute={m['compute']:.4f} batch={m['batch']:.4f} "
            f"accuracy={m['accuracy']:.4f} stop_mean={m['stop_mean']:.4f}"
        ),
        flush=True,
    )


def train_main(cfg: LMMSConfig) -> None:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    vocab = extend_tokenizer(tokenizer, vz=cfg.model.Vz)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model.resize_token_embeddings(len(tokenizer))
    base_model = base_model.to(dtype=torch.bfloat16)

    model = LMMSWrapper(
        base_model=base_model,
        vocab=vocab,
        kmax=cfg.model.Kmax,
        vz=cfg.model.Vz,
    ).to(device=device, dtype=torch.bfloat16)

    train_ds = LMMSDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        tokenizer=tokenizer,
        vocab=vocab,
        kmax=cfg.model.Kmax,
        max_length=cfg.data.max_length,
    )
    eval_ds = LMMSDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.eval_split,
        tokenizer=tokenizer,
        vocab=vocab,
        kmax=cfg.model.Kmax,
        max_length=cfg.data.max_length,
    )

    collator = LMMScollator(pad_token_id=tokenizer.pad_token_id, kmax=cfg.model.Kmax)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
    )

    opt = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    cfg_dump_path = os.path.join(cfg.train.output_dir, "run_config.json")
    with open(cfg_dump_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Training on device={device}", flush=True)
    step_iter = _iter_forever(train_loader)
    artifact_every = 0
    if cfg.train.eval_every > 0 and cfg.train.eval_generate_every_mult > 0:
        artifact_every = cfg.train.eval_every * cfg.train.eval_generate_every_mult

    for step in range(1, cfg.train.num_train_steps + 1):
        model.train()
        model.base_model.config.use_cache = False
        batch = _move_batch_to_device(next(step_iter), device)

        opt.zero_grad(set_to_none=True)
        total, train_metrics = run_three_pass(
            model=model,
            batch=batch,
            cfg=cfg,
            deterministic=False,
            global_step=step,
        )
        total.backward()
        opt.step()

        if cfg.train.print_every > 0 and step % cfg.train.print_every == 0:
            _print_metrics("train", step, train_metrics)

        if cfg.train.eval_every > 0 and step % cfg.train.eval_every == 0:
            eval_metrics = evaluate(model, eval_loader, cfg=cfg, device=device)
            _print_metrics("eval", step, eval_metrics)

        if artifact_every > 0 and step % artifact_every == 0:
            artifact_path = generate_eval_artifact(
                model=model,
                tokenizer=tokenizer,
                vocab=vocab,
                eval_dataset=eval_ds,
                cfg=cfg,
                device=device,
                global_step=step,
            )
            print(f"[artifact] step={step} path={artifact_path}", flush=True)

    ckpt_path = os.path.join(cfg.train.output_dir, "lmms_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)
