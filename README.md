# LMMS – Latent Markov Math Solver (Discrete Z + Dynamic K)

This phase converts a pretrained decoder-only LLM into a **Latent Markov Math Solver (LMMS)** by introducing:

- A discrete latent action space `Z`
- A stop action (`<ANSWER>`) inside the latent loop
- Dynamic compute via learned stop-time (**no dataset-provided K**)
- Counterfactual supervision enforcing **causal dependence on latent states**

Unlike previous versions:

1. No Coconut (continuous latent pretraining)
2. No dataset-provided K
3. Stop-time and compute budget are learned via loss

---

# 1. High-Level Idea

We transform a standard LLM into a model that:

1. Receives a math question.
2. Executes up to `Kmax` discrete latent steps.
3. Stops early by emitting `<ANSWER>` as a latent action.
4. Produces a numeric answer via a 5-digit head.

The latent sequence forms a **Markovian internal program**, not natural-language chain-of-thought.

This prepares the model for PPO where:

- `Z` tokens become actions
- `<ANSWER>` becomes termination
- Reward = correctness − compute cost

---

## 1.1 Definitions

| Symbol | Meaning |
|--------|----------|
| B | Batch size |
| T | Sequence length |
| H | Hidden size |
| Kmax | Maximum latent slots |
| Vz | Latent vocabulary size |
| t_stop | First latent index where `<ANSWER>` is sampled |
| alive slot | t ≤ t_stop |
| dead slot | t > t_stop |

---

# 2. Token Design

We extend the tokenizer with:

- `<|latent|>` – placeholder slots
- `<ANSWER>` – stop action + readout anchor
- `<Z_0> ... <Z_{Vz-1}>` – discrete latent vocabulary

## Injection Rules

At latent slot `t`, action `A_t`:

- If `A_t = <Z_i>` → inject embedding `E(<Z_i>)`
- If `A_t = <ANSWER>` at first stop → inject `E(<ANSWER>)`, later slots become dead
- Dead slots → inject zero vector (or mask attention)

## Two Roles of `<ANSWER>`

- **Role A**: Stop action in latent policy
- **Role B**: Fixed readout anchor token (always present after latent slots)

Final digits are predicted **only from the hidden state at the fixed `<ANSWER>` position**.

---

# 3. Dataset

We use the HuggingFace dataset:

omrisap/phaseZ


## Splits

- `train`: 50,000 samples
- `eval`: 1,000 samples

## Used Columns

We only use:

- `question: str`
- `answer_digits: int[5]` (always length 5)

All other dataset columns are ignored.

---

# 4. Prompting Contract

We wrap questions in a chat template when available.

## If chat template exists

system: You are a math reasoning model. Return only the final numeric answer.
user: <question>


## Fallback

Solve the following math problem. Return only the final numeric answer.
Question: <question>


---

# 5. Input Format (Training)

Training input sequence:

[prompt(question)] + <|latent|> * Kmax + <ANSWER>


Example (`Kmax=4`):

What is 37 * 12 ? <|latent|> <|latent|> <|latent|> <|latent|> <ANSWER>


---

# 6. Dynamic-K Semantics

At each latent slot:

A_t ∈ (Z ∪ {<ANSWER>})


Rules:

- If `A_t ∈ Z` → continue
- If `A_t = <ANSWER>` → stop, later slots dead
- If no stop occurs before `Kmax-1`, force stop at last slot

The first `<ANSWER>` slot is considered alive.

---

# 7. Model Architecture

Start from `AutoModelForCausalLM`.

Add:

- New tokenizer tokens
- Resize embeddings
- 5 digit heads: `Linear(H, 10)`

---

## 7.1 Latent Logits via LM Head (Restricted Geometry)

We do **not** add a separate latent head.

Instead:

Let `W_lm` be the LM head weight matrix.

For hidden state `h_t`:

logits_Z = h_t @ W_lm[z_token_ids].T
logits_stop = h_t @ W_lm[answer_token_id].T
logits = concat(logits_Z, logits_stop)


We do not train with full-vocabulary next-token CE.

---

# 8. Training Algorithm

Three-pass supervised flow:

1. Pass 1 – policy proposal
2. Pass 2 – reference injection
3. Pass 3 – counterfactual injection

---

## 8.1 Pass 1 (Single Forward Policy Proposal)

We run one transformer forward:

[prompt] + <|latent|>*Kmax + <ANSWER>


We compute:

logits_all ∈ [B, Kmax, Vz+1]
p_stop ∈ [B, Kmax]


Actions are sampled using GS-ST.

### GS-ST (Gumbel-Softmax Straight-Through)

- Hard sample in forward
- Soft gradient in backward
- Enables discrete actions with gradients

We add Gumbel noise to latent logits before softmax, then apply straight-through hard sampling.

Forced stop is applied if needed.

Alive mask is computed from first stop.

---

## 8.2 Pass 2 (Reference Injection)

Inject sampled actions.

Forward pass.

Extract hidden state at fixed `<ANSWER>`.

Predict digits.

---

## 8.3 Pass 3 (Counterfactual Injection)

Apply one perturbation mode:

- Random replacement
- Permutation
- Truncation

Re-run forward pass.

Compute JS divergence.

---

# 9. Losses

We train with four losses.

## 9.1 Answer Loss

Digit-wise cross entropy over 5 digits.

To mitigate digit collapse (e.g., trivial all-zero prediction or overconfidence in leading zeros), we optionally apply digit supervision dropout.

Let:

keep_prob[i] be the probability of keeping supervision for digit position i.

For digit index i ∈ {0..4}:

L_i = CE(logits_i, target_i)


If keep_prob is enabled:

mask_i ~ Bernoulli(keep_prob[i])
L_i = mask_i * CE(...)


This prevents the model from overfitting to easy digit positions (e.g., frequent leading zeros) and encourages learning across all digits.

If keep_prob=None, full supervision is applied to all digits.

---

## 9.2 Counterfactual Loss

L_cf = log(2) − JS(p_ref, p_cf)


Encourages latent causal dependence.

---

## 9.3 Compute Loss

Let:

S[0] = 1
S[t] = ∏_{i=0..t-1}(1 − p_stop[i])

We use `S[t]` as the survival probability before step `t` (0-indexed).


Then:

L_compute = λ_compute * Σ_{t=0..Kmax-1} w_t * S[t]


Uses stop probabilities before forced termination masking to preserve gradient signal.

---

## 9.4 Batch Collision Loss

Herfindahl penalty:

L_batch = Σ p_v²


---

## 9.5 Total

L_total = w_answer L_answer
+ w_cf L_cf
+ w_compute L_compute
+ w_batch L_batch


---

## 9.6 Digit Supervision Dropout (Optional)

In practice, fixed-length numeric answers often contain structured bias (e.g., leading zeros).

Without regularization, the model may:

Predict trivial zero-heavy outputs

Rely excessively on prompt-only shortcuts

Under-utilize latent reasoning steps

Digit supervision dropout mitigates this by:

Reducing dominance of easy digit positions

Encouraging gradient flow across all digits

Making counterfactual and latent losses more influential

This mechanism is optional and primarily used as a stabilization or anti-collapse tool.

Digit supervision dropout only affects the answer head loss. It does not modify latent policy logits or stop-time learning.

---

# 10. Deterministic Evaluation

Evaluation uses:

- Argmax latent policy
- No sampling
- No stochasticity

Metrics printed:

total, answer, cf, compute, batch, accuracy, stop_mean

Evaluation always uses argmax (no temperature); temperature is used only in training-time latent sampling and sampled generation artifacts.


---

# 11. Training & Evaluation Schedule

Let:

- `x` = train log interval
- `y` = eval interval
- `z` = prediction artifact interval
- `z = train.eval_every * train.eval_generate_every_mult`

### Every `x` steps
Print training losses.

### Every `y` steps
Run deterministic eval on eval split.

### Every `z` steps
Generate prediction artifact.

---

# 12. Prediction Artifact (Generation-Based Evaluation)

Every `z` steps we generate predictions on eval split.
Concretely, artifact writes happen every `train.eval_every * train.eval_generate_every_mult` training steps.
Training uses explicit latent slot injection and multi-pass execution, while generation evaluates whether latent reasoning is internalized into standard autoregressive decoding.

Important:

- We do NOT append `<|latent|>` slots during generation.
- Model receives only prompt.
- Generation stops when `<ANSWER>` is generated (acts as EOS) or at `max_tokens`.
- During generation, `<Z_i>` and `<ANSWER>` are generated autoregressively like normal tokens; no placeholder injection or multi-pass reruns are used.

---

## Saved Fields Per Example

Each JSON entry:

Digit predictions in artifact runs are computed using the digit heads from the hidden state at the generated `<ANSWER>` position.

```json
{
  "question": "...",
  "answer_digits": [0,1,2,3,4],

  "greedy_full_text": "...",
  "greedy_digit_pred": [0,1,2,3,4] | null,

  "sample_full_text": "...",
  "sample_digit_pred": [0,1,2,3,4] | null,

  "greedy_randomized_full_text": "...",
  "greedy_randomized_digit_pred": [0,1,2,3,4] | null,

  "sample_randomized_full_text": "...",
  "sample_randomized_digit_pred": [0,1,2,3,4] | null,

  "greedy_truncated_full_text": "...",
  "greedy_truncated_digit_pred": [0,1,2,3,4] | null,

  "sample_truncated_full_text": "...",
  "sample_truncated_digit_pred": [0,1,2,3,4] | null
}
```

## Randomized-Z Ablation

Replace each generated <Z_i> with random <Z_j>.

Recompute digit prediction.

## Truncated-Z Ablation

Truncate generated Z sequence before <ANSWER>.

Recompute digit prediction.

# 13. Config Defaults

| Name | Default |
|------|---------|
| model_name_or_path | Qwen/Qwen2.5-Math-1.5B |
| Kmax | 16 |
| Vz | 512 |
| gamma | 1.5 |
| lambda_compute | 1.0 |
| w_answer | 1.0 |
| w_cf | 1.0 |
| w_compute | 0.1 |
| w_batch | 0.01 |
| tau | 1.0 |

## 13.1 Extra Behavior Knobs (Not In Defaults Table)

These knobs materially affect training/eval behavior. Status is explicitly called out.

### Train/Eval cadence

- `train.print_every`: **Used in training**; defines `x` (see Section 11).
- `train.eval_every`: **Used in training**; defines `y` (see Section 11).
- `train.eval_generate_every_mult`: **Used in training**; defines `z = train.eval_every * train.eval_generate_every_mult` (see Sections 11 and 12).

### Generation artifact controls

- `train.eval_generate_max_new_tokens`: **Used in training** for artifact generation limit (`max_tokens`) (see Section 12).
- `train.eval_generate_temperature`: **Used in training** for sampled artifact generations (see Section 12).
- `train.eval_generate_top_p`: **Used in training** for sampled artifact generations (see Section 12).

### Counterfactual warmup / unfreeze and biasing

- `train.cf_warmup_steps`: **Used in training**; delays full counterfactual-loss pressure early in training (see Sections 8 and 9.2).
- `train.cf_bias_anneal_steps`: **Legacy-only (not used in current phase)**.
- `train.cf_attention_bias_strength`: **Legacy-only (not used in current phase)**.
- `train.cf_attention_bias_enabled`: **Legacy-only (not used in current phase)**.
- `train.cf_bias_apply_cf_path_only`: **Legacy-only (not used in current phase)**.

### Dataset handling

- `data.batch_size`: **Used in training** (batch semantics; see Section 1.1 `B` and training loop behavior).
- `data.max_length`: **Used in training** (prompt/context truncation behavior).
- `data.rebalance_train`: **Legacy-only (not used in current phase)**.

### Objective-affecting extras

- `loss.lambda_sft`: **Legacy-only (not used in current phase)**.
- `loss.lambda_no_answer_on_latent`: **Legacy-only (not used in current phase)**.
- `loss.digit_temperature`: **Legacy-only (not used in current phase)**.
- `loss.keep_prob`: Optional digit supervision dropout probabilities.
  Example:
  `(0.02, 0.05, 0.1, 0.5, 1.0)`
  When set, digit-wise cross entropy is randomly masked per position according to these probabilities.
  Default: None (disabled).

# 14. Quickstart

```bash
python train.py \
  --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
  --dataset_name omrisap/phaseZ \
  --Kmax 16 \
  --Vz 512
```

# 15. PPO-Readiness

This phase is PPO-ready:

State = hidden before latent decision

Action = Z ∪ {<ANSWER>}

Termination = <ANSWER>

Compute penalty already defined
