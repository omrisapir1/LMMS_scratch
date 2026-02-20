from __future__ import annotations

from dataclasses import dataclass
import argparse


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B"
    Kmax: int = 32
    Vz: int = 512
    tau: float = 1.0


@dataclass
class DataConfig:
    dataset_name: str = "omrisap/phaseZ"
    train_split: str = "train"
    eval_split: str = "eval"
    batch_size: int = 64
    max_length: int = 512
    num_workers: int = 0


@dataclass
class LossConfig:
    gamma: float = 1.5
    w_cf: float = 1.0
    w_batch: float = 0.2
    # warmup_* overrides apply for steps < warmup_steps
    warmup_w_answer: float = 0.03
    warmup_w_cf: float = 1.0
    warmup_w_compute: float = 0.0
    warmup_w_batch: float = 1.0
    # ---- Answer weight annealing ----
    w_answer_start: float = 0.03
    w_answer_end: float = 1.0
    w_answer_anneal_steps: int = 1000
    # ---- Compute weight annealing ----
    w_compute_start: float = 0.01
    w_compute_end: float = 0.1
    w_compute_anneal_steps: int = 2000
    keep_prob: tuple[float, ...] = (0.05, 0.1, 0.15, 0.75, 1.0)


@dataclass
class TrainConfig:
    seed: int = 42
    lr: float = 3e-5
    # warmup_* overrides apply for steps < warmup_steps
    warmup_steps: int = 100
    warmup_lr: float = 3e-4
    weight_decay: float = 0.0
    num_train_steps: int = 2000
    print_every: int = 20
    eval_every: int = 100
    eval_generate_every_mult: int = 10
    eval_generate_max_new_tokens: int = 128
    eval_generate_temperature: float = 1.0
    eval_generate_top_p: float = 0.95
    cf_warmup_steps: int = 200
    device: str = "cuda"
    output_dir: str = "outputs"


@dataclass
class LMMSConfig:
    model: ModelConfig
    data: DataConfig
    loss: LossConfig
    train: TrainConfig


def _parse_keep_prob(raw: str | None, parser: argparse.ArgumentParser) -> tuple[float, ...] | None:
    if raw is None:
        return None

    text = raw.strip()
    if text == "":
        return None

    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 5:
        parser.error("--keep_prob must have exactly 5 comma-separated floats, e.g. 0.02,0.05,0.1,0.5,1.0")

    vals = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            parser.error(f"Invalid keep_prob value: {p}")
        if v < 0.0 or v > 1.0:
            parser.error(f"keep_prob values must be in [0,1], got {v}")
        vals.append(v)

    return tuple(vals)


def parse_args() -> LMMSConfig:
    parser = argparse.ArgumentParser(description="LMMS training")

    parser.add_argument("--model_name_or_path", type=str, default=ModelConfig.model_name_or_path)
    parser.add_argument("--Kmax", type=int, default=ModelConfig.Kmax)
    parser.add_argument("--Vz", type=int, default=ModelConfig.Vz)
    parser.add_argument("--tau", type=float, default=ModelConfig.tau)

    parser.add_argument("--dataset_name", type=str, default=DataConfig.dataset_name)
    parser.add_argument("--train_split", type=str, default=DataConfig.train_split)
    parser.add_argument("--eval_split", type=str, default=DataConfig.eval_split)
    parser.add_argument("--batch_size", type=int, default=DataConfig.batch_size)
    parser.add_argument("--max_length", type=int, default=DataConfig.max_length)
    parser.add_argument("--num_workers", type=int, default=DataConfig.num_workers)

    parser.add_argument("--gamma", type=float, default=LossConfig.gamma)
    parser.add_argument("--warmup_w_answer", type=float, default=LossConfig.warmup_w_answer)
    parser.add_argument("--w_answer_start", type=float, default=LossConfig.w_answer_start)
    parser.add_argument("--w_answer_end", type=float, default=LossConfig.w_answer_end)
    parser.add_argument("--w_answer_anneal_steps", type=int, default=LossConfig.w_answer_anneal_steps)
    parser.add_argument("--w_cf", type=float, default=LossConfig.w_cf)
    parser.add_argument("--warmup_w_cf", type=float, default=LossConfig.warmup_w_cf)
    parser.add_argument("--warmup_w_compute", type=float, default=LossConfig.warmup_w_compute)
    parser.add_argument("--w_compute_start", type=float, default=LossConfig.w_compute_start)
    parser.add_argument("--w_compute_end", type=float, default=LossConfig.w_compute_end)
    parser.add_argument("--w_compute_anneal_steps", type=int, default=LossConfig.w_compute_anneal_steps)
    parser.add_argument("--w_batch", type=float, default=LossConfig.w_batch)
    parser.add_argument("--warmup_w_batch", type=float, default=LossConfig.warmup_w_batch)
    parser.add_argument(
        "--keep_prob",
        type=str,
        default=None,
        help="Comma-separated keep probabilities for 5 digit positions, e.g. 0.02,0.05,0.1,0.5,1.0",
    )

    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--warmup_lr", type=float, default=TrainConfig.warmup_lr)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num_train_steps", type=int, default=TrainConfig.num_train_steps)
    parser.add_argument("--print_every", type=int, default=TrainConfig.print_every)
    parser.add_argument("--eval_every", type=int, default=TrainConfig.eval_every)
    parser.add_argument(
        "--eval_generate_every_mult",
        type=int,
        default=TrainConfig.eval_generate_every_mult,
    )
    parser.add_argument(
        "--eval_generate_max_new_tokens",
        type=int,
        default=TrainConfig.eval_generate_max_new_tokens,
    )
    parser.add_argument(
        "--eval_generate_temperature",
        type=float,
        default=TrainConfig.eval_generate_temperature,
    )
    parser.add_argument(
        "--eval_generate_top_p",
        type=float,
        default=TrainConfig.eval_generate_top_p,
    )
    parser.add_argument("--cf_warmup_steps", type=int, default=TrainConfig.cf_warmup_steps)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)

    args = parser.parse_args()
    keep_prob = _parse_keep_prob(args.keep_prob, parser)

    return LMMSConfig(
        model=ModelConfig(
            model_name_or_path=args.model_name_or_path,
            Kmax=args.Kmax,
            Vz=args.Vz,
            tau=args.tau,
        ),
        data=DataConfig(
            dataset_name=args.dataset_name,
            train_split=args.train_split,
            eval_split=args.eval_split,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
        ),
        loss=LossConfig(
            gamma=args.gamma,
            warmup_w_answer=args.warmup_w_answer,
            w_answer_start=args.w_answer_start,
            w_answer_end=args.w_answer_end,
            w_answer_anneal_steps=args.w_answer_anneal_steps,
            w_cf=args.w_cf,
            warmup_w_cf=args.warmup_w_cf,
            warmup_w_compute=args.warmup_w_compute,
            w_compute_start=args.w_compute_start,
            w_compute_end=args.w_compute_end,
            w_compute_anneal_steps=args.w_compute_anneal_steps,
            w_batch=args.w_batch,
            warmup_w_batch=args.warmup_w_batch,
            keep_prob=keep_prob,
        ),
        train=TrainConfig(
            seed=args.seed,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            warmup_lr=args.warmup_lr,
            weight_decay=args.weight_decay,
            num_train_steps=args.num_train_steps,
            print_every=args.print_every,
            eval_every=args.eval_every,
            eval_generate_every_mult=args.eval_generate_every_mult,
            eval_generate_max_new_tokens=args.eval_generate_max_new_tokens,
            eval_generate_temperature=args.eval_generate_temperature,
            eval_generate_top_p=args.eval_generate_top_p,
            cf_warmup_steps=args.cf_warmup_steps,
            device=args.device,
            output_dir=args.output_dir,
        ),
    )
