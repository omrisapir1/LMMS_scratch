from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class LMMSVocab:
    latent_placeholder: str
    answer_token: str
    z_tokens: List[str]
    latent_placeholder_id: int
    answer_token_id: int
    z_token_ids: List[int]

    @property
    def action_tokens(self) -> List[str]:
        return [*self.z_tokens, self.answer_token]

    @property
    def action_token_ids(self) -> List[int]:
        return [*self.z_token_ids, self.answer_token_id]


SYSTEM_PROMPT = "You are a math reasoning model. Return only the final numeric answer."


def extend_tokenizer(tokenizer, vz: int) -> LMMSVocab:
    latent_placeholder = "<|latent|>"
    answer_token = "<ANSWER>"
    z_tokens = [f"<Z_{i}>" for i in range(vz)]

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [latent_placeholder, answer_token, *z_tokens],
        }
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    latent_placeholder_id = tokenizer.convert_tokens_to_ids(latent_placeholder)
    answer_token_id = tokenizer.convert_tokens_to_ids(answer_token)
    z_token_ids = tokenizer.convert_tokens_to_ids(z_tokens)

    return LMMSVocab(
        latent_placeholder=latent_placeholder,
        answer_token=answer_token,
        z_tokens=z_tokens,
        latent_placeholder_id=latent_placeholder_id,
        answer_token_id=answer_token_id,
        z_token_ids=z_token_ids,
    )


def format_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

