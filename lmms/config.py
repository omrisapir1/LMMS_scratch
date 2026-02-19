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
    lambda_compute: float = 1.0
    w_answer: float = 1.0
    w_cf: float = 1.0
    w_compute: float = 0.1
    w_batch: float = 0.01


@dataclass
class TrainConfig:
    seed: int = 42
    lr: float = 1e-5
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
    artifact_eval_examples: int = 64


@dataclass
class LMMSConfig:
    model: ModelConfig
    data: DataConfig
    loss: LossConfig
    train: TrainConfig


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
    parser.add_argument("--lambda_compute", type=float, default=LossConfig.lambda_compute)
    parser.add_argument("--w_answer", type=float, default=LossConfig.w_answer)
    parser.add_argument("--w_cf", type=float, default=LossConfig.w_cf)
    parser.add_argument("--w_compute", type=float, default=LossConfig.w_compute)
    parser.add_argument("--w_batch", type=float, default=LossConfig.w_batch)

    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
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
    parser.add_argument(
        "--artifact_eval_examples",
        type=int,
        default=TrainConfig.artifact_eval_examples,
    )

    args = parser.parse_args()
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
            lambda_compute=args.lambda_compute,
            w_answer=args.w_answer,
            w_cf=args.w_cf,
            w_compute=args.w_compute,
            w_batch=args.w_batch,
        ),
        train=TrainConfig(
            seed=args.seed,
            lr=args.lr,
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
            artifact_eval_examples=args.artifact_eval_examples,
        ),
    )
