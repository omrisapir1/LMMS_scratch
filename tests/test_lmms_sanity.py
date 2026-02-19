import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from unittest.mock import patch

from lmms.losses import batch_collision_loss, compute_loss_unmasked_stop, counterfactual_loss
from lmms.model import LMMSWrapper
from lmms.tokenizer import LMMSVocab, extend_tokenizer


def _build_wrapper(kmax: int = 4, vz: int = 3, vocab_size: int = 64) -> LMMSWrapper:
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=1,
        n_head=2,
    )
    base = GPT2LMHeadModel(cfg)
    vocab = LMMSVocab(
        latent_placeholder="<|latent|>",
        answer_token="<ANSWER>",
        z_tokens=[f"<Z_{i}>" for i in range(vz)],
        latent_placeholder_id=5,
        answer_token_id=9,
        z_token_ids=list(range(6, 6 + vz)),
    )
    return LMMSWrapper(base_model=base, vocab=vocab, kmax=kmax, vz=vz)


class _DummyTokenizer:
    def __init__(self):
        self.vocab = {"<eos>": 0}
        self.additional_special_tokens = []
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self._pad_token = None
        self.pad_token_id = None
        self.chat_template = None

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, tok):
        self._pad_token = tok
        self.pad_token_id = self.vocab[tok]

    def add_special_tokens(self, mapping):
        added = 0
        for tok in mapping.get("additional_special_tokens", []):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
                self.additional_special_tokens.append(tok)
                added += 1
        if "pad_token" in mapping:
            tok = mapping["pad_token"]
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
                added += 1
            self.pad_token = tok
        return added

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.vocab[t] for t in tokens]
        return self.vocab[tokens]

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def special_tokens_map(self):
        out = {"eos_token": self.eos_token}
        if self.pad_token is not None:
            out["pad_token"] = self.pad_token
        if self.additional_special_tokens:
            out["additional_special_tokens"] = list(self.additional_special_tokens)
        return out

    def add_tokens(self, new_tokens):
        added = 0
        for tok in new_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
                added += 1
        return added


def test_extend_tokenizer_adds_lmms_tokens():
    tok = _DummyTokenizer()
    vocab = extend_tokenizer(tok, vz=4)
    assert vocab.latent_placeholder in tok.vocab
    assert vocab.answer_token in tok.vocab
    assert len(vocab.z_token_ids) == 4
    assert len(set(vocab.action_token_ids)) == 5
    assert vocab.action_token_ids[-1] == vocab.answer_token_id
    assert vocab.action_token_ids[:-1] == vocab.z_token_ids
    assert tok.pad_token_id == tok.eos_token_id


def test_force_stop_and_alive_mask():
    model = _build_wrapper(kmax=4, vz=3)

    hard = torch.zeros(1, 4, 4)
    hard[:, :, 0] = 1.0  # never stop
    out = model._force_stop(hard)

    action_ids = out["action_ids"]
    alive = out["alive_mask"]
    first_stop = out["first_stop"]

    assert action_ids[0, -1].item() == model.stop_action_index
    assert first_stop[0].item() == 3
    assert alive[0].tolist() == [True, True, True, True]

    hard2 = torch.zeros(1, 4, 4)
    hard2[0, 0, 1] = 1.0
    hard2[0, 1, model.stop_action_index] = 1.0
    hard2[0, 2, 2] = 1.0
    hard2[0, 3, 0] = 1.0

    out2 = model._force_stop(hard2)
    assert out2["first_stop"][0].item() == 1
    assert out2["alive_mask"][0].tolist() == [True, True, False, False]


def test_batch_collision_alive_z_only_herfindahl():
    action_ids = torch.tensor(
        [
            [0, 1, 3, 2],
            [1, 3, 2, 0],
        ],
        dtype=torch.long,
    )
    alive_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    # Valid alive non-stop actions are [0, 1, 1].
    loss = batch_collision_loss(
        action_ids=action_ids,
        alive_mask=alive_mask,
        stop_action_index=3,
        vz=3,
    )
    expected = (1.0 / 3.0) ** 2 + (2.0 / 3.0) ** 2
    assert abs(loss.item() - expected) < 2e-3


def test_counterfactual_detaches_reference_branch():
    ref = torch.randn(2, 5, 10, requires_grad=True)
    cf = torch.randn(2, 5, 10, requires_grad=True)
    loss = counterfactual_loss(ref, cf)
    loss.backward()
    assert ref.grad is None or ref.grad.abs().sum().item() == 0.0
    assert cf.grad is not None


def test_pass_shapes_and_digit_head_shapes():
    torch.manual_seed(0)
    model = _build_wrapper(kmax=4, vz=3)

    # Sequence layout: prompt(3) + Kmax(4) + fixed <ANSWER>(1)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 5, 5, 5, 5, 9],
            [4, 2, 1, 5, 5, 5, 5, 9],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    latent_positions = torch.tensor(
        [
            [3, 4, 5, 6],
            [3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    answer_positions = torch.tensor([7, 7], dtype=torch.long)

    pass1 = model.pass1_policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_positions=latent_positions,
        tau=1.0,
        deterministic=True,
    )

    assert pass1.logits_all.shape == (2, 4, 4)
    assert pass1.p_stop_unmasked.shape == (2, 4)
    assert pass1.action_probs.shape == (2, 4, 4)
    assert pass1.alive_mask.shape == (2, 4)

    ref_hidden = model.pass_with_injection(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_positions=latent_positions,
        action_probs=pass1.action_probs,
        alive_mask=pass1.alive_mask,
    )
    ans_hidden = model.extract_answer_hidden(ref_hidden, answer_positions)
    digit_logits = model.digit_logits_from_hidden(ans_hidden)

    assert ref_hidden.shape[:2] == input_ids.shape
    assert ans_hidden.shape == (2, model._hidden_size())
    assert digit_logits.shape == (2, 5, 10)


def test_pass1_policy_single_transformer_forward():
    model = _build_wrapper(kmax=4, vz=3)
    input_ids = torch.tensor([[1, 2, 3, 5, 5, 5, 5, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    latent_positions = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

    with patch.object(
        model.base_model.base_model,
        "forward",
        wraps=model.base_model.base_model.forward,
    ) as mocked_forward:
        _ = model.pass1_policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_positions=latent_positions,
            tau=1.0,
            deterministic=True,
        )
        assert mocked_forward.call_count == 1


def test_pass_with_injection_applies_dead_slot_attention_mask():
    model = _build_wrapper(kmax=4, vz=3)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 5, 5, 5, 5, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    latent_positions = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    alive_mask = torch.tensor([[True, True, False, False]])
    action_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    action_probs = F.one_hot(action_ids, num_classes=4).float()

    expected_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 1]], dtype=torch.long)
    with patch.object(
        model.base_model.base_model,
        "forward",
        wraps=model.base_model.base_model.forward,
    ) as mocked_forward:
        with torch.no_grad():
            _ = model.pass_with_injection(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_positions=latent_positions,
                action_probs=action_probs,
                alive_mask=alive_mask,
            )

    assert mocked_forward.call_count == 1
    call = mocked_forward.call_args
    if "attention_mask" in call.kwargs:
        used_mask = call.kwargs["attention_mask"]
    else:
        # GPT-style forward signature: (input_ids, past_key_values, attention_mask, ...)
        if len(call.args) > 2:
            used_mask = call.args[2]
        else:
            raise AssertionError("attention_mask not found in transformer forward call")
    assert torch.equal(used_mask, expected_mask)


def test_pass_with_injection_dead_actions_do_not_change_answer_hidden():
    model = _build_wrapper(kmax=4, vz=3)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 5, 5, 5, 5, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    latent_positions = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    answer_positions = torch.tensor([7], dtype=torch.long)
    alive_mask = torch.tensor([[True, True, False, False]])

    action_ids_a = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    action_ids_b = torch.tensor([[0, 1, 0, 2]], dtype=torch.long)
    action_probs_a = F.one_hot(action_ids_a, num_classes=4).float()
    action_probs_b = F.one_hot(action_ids_b, num_classes=4).float()

    with torch.no_grad():
        h_a = model.pass_with_injection(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_positions=latent_positions,
            action_probs=action_probs_a,
            alive_mask=alive_mask,
        )
        h_b = model.pass_with_injection(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_positions=latent_positions,
            action_probs=action_probs_b,
            alive_mask=alive_mask,
        )
        ans_a = model.extract_answer_hidden(h_a, answer_positions)
        ans_b = model.extract_answer_hidden(h_b, answer_positions)

    assert torch.allclose(ans_a, ans_b, atol=1e-6, rtol=0.0)


def test_deterministic_pass1_uses_argmax_policy():
    model = _build_wrapper(kmax=4, vz=3)
    hidden_size = model._hidden_size()
    input_ids = torch.tensor([[1, 2, 3, 5, 5, 5, 5, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    latent_positions = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

    # Shape [B, K, Vz+1], stop index is 3. Argmax per slot => [0, 3, 1, 2].
    logits = torch.tensor(
        [
            [
                [9.0, 1.0, 0.0, -1.0],
                [0.0, 0.1, 0.2, 4.0],
                [0.0, 2.0, 1.0, -5.0],
                [0.1, -0.2, 0.3, -0.4],
            ]
        ],
        dtype=torch.float32,
    )

    with patch.object(model, "forward_hidden", return_value=torch.zeros(1, 8, hidden_size)):
        with patch.object(model, "latent_logits_from_hidden", return_value=logits):
            out_tau_low = model.pass1_policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_positions=latent_positions,
                tau=0.01,
                deterministic=True,
            )
            out_tau_high = model.pass1_policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_positions=latent_positions,
                tau=100.0,
                deterministic=True,
            )

    expected_ids = torch.tensor([[0, 3, 1, 2]], dtype=torch.long)
    expected_alive = torch.tensor([[True, True, False, False]])
    expected_first_stop = torch.tensor([1], dtype=torch.long)
    assert torch.equal(out_tau_low.action_ids, expected_ids)
    assert torch.equal(out_tau_high.action_ids, expected_ids)
    assert torch.equal(out_tau_low.alive_mask, expected_alive)
    assert torch.equal(out_tau_high.alive_mask, expected_alive)
    assert torch.equal(out_tau_low.first_stop, expected_first_stop)
    assert torch.equal(out_tau_high.first_stop, expected_first_stop)


def test_dead_latent_slots_are_masked_in_attention():
    model = _build_wrapper(kmax=4, vz=3)
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    latent_positions = torch.tensor(
        [
            [3, 4, 5, 6],
            [3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    alive_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
        ]
    )

    masked = model.mask_dead_latent_attention(
        attention_mask=attention_mask,
        latent_positions=latent_positions,
        alive_mask=alive_mask,
    )

    assert masked[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 1]
    assert masked[1].tolist() == [1, 1, 1, 1, 0, 0, 0, 1]
