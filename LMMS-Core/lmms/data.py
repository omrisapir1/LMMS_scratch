from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .tokenizer import LMMSVocab, format_prompt


@dataclass
class EncodedExample:
    input_ids: List[int]
    attention_mask: List[int]
    latent_positions: List[int]
    answer_position: int
    answer_digits: List[int]
    prompt_ids: List[int]


def _encode_single(
    tokenizer,
    vocab: LMMSVocab,
    question: str,
    answer_digits: List[int],
    kmax: int,
    max_length: int,
) -> EncodedExample:
    if len(answer_digits) != 5:
        raise ValueError(f"answer_digits must be length 5, got {len(answer_digits)}")

    if max_length <= kmax + 1:
        raise ValueError("max_length must be > Kmax + 1 to leave room for prompt")

    prompt_text = format_prompt(tokenizer, question)
    max_prompt_len = max_length - (kmax + 1)
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_prompt_len,
    )["input_ids"]

    latent_positions = list(range(len(prompt_ids), len(prompt_ids) + kmax))
    answer_position = len(prompt_ids) + kmax

    input_ids = [
        *prompt_ids,
        *([vocab.latent_placeholder_id] * kmax),
        vocab.answer_token_id,
    ]
    attention_mask = [1] * len(input_ids)

    return EncodedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_positions=latent_positions,
        answer_position=answer_position,
        answer_digits=[int(x) for x in answer_digits],
        prompt_ids=prompt_ids,
    )


class LMMSDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        vocab: LMMSVocab,
        kmax: int,
        max_length: int,
    ):
        self.ds = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.kmax = kmax
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.ds[idx]
        ex = _encode_single(
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            question=row["question"],
            answer_digits=row["answer_digits"],
            kmax=self.kmax,
            max_length=self.max_length,
        )
        return {
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "latent_positions": ex.latent_positions,
            "answer_position": ex.answer_position,
            "answer_digits": ex.answer_digits,
            "prompt_ids": ex.prompt_ids,
            "question": row["question"],
        }


class LMMScollator:
    def __init__(self, pad_token_id: int, kmax: int):
        self.pad_token_id = pad_token_id
        self.kmax = kmax

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor | List[str] | List[List[int]]]:
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        latent_positions = []
        answer_positions = []
        answer_digits = []
        prompt_ids = []
        questions = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            latent_positions.append(item["latent_positions"])
            answer_positions.append(item["answer_position"])
            answer_digits.append(item["answer_digits"])
            prompt_ids.append(item["prompt_ids"])
            questions.append(item["question"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "latent_positions": torch.tensor(latent_positions, dtype=torch.long),
            "answer_positions": torch.tensor(answer_positions, dtype=torch.long),
            "answer_digits": torch.tensor(answer_digits, dtype=torch.long),
            "prompt_ids": prompt_ids,
            "questions": questions,
        }
