# Embedding, Attention Head & Embedding Dimension Analysis

## Embedding Dimension (`n_embd = 768`)

Defined in [train.py:54](train.py#L54) and [model.py:114](model.py#L114), this single value threads through the entire model.

### Token Embedding — [model.py:127](model.py#L127)

```python
wte = nn.Embedding(vocab_size, n_embd)   # shape: (vocab_size, 768)
```

Each token in the vocabulary gets a 768-dimensional vector.

### Positional Embedding — [model.py:128](model.py#L128)

```python
wpe = nn.Embedding(block_size, n_embd)   # shape: (1024, 768)
```

Each position (up to 1024) also gets a 768-dim learned vector. These two are **summed** before the transformer blocks ([model.py:179](model.py#L179)).

### Weight Tying — [model.py:138](model.py#L138)

The token embedding matrix `wte` is shared with the output `lm_head` linear layer. This saves ~38M parameters and is a standard GPT-2 trick.

---

## Attention Heads (`n_head = 12`)

Defined in [train.py:53](train.py#L53) and [model.py:113](model.py#L113).

### Head Dimension

Each head operates on a slice of size:

```
head_size = n_embd / n_head = 768 / 12 = 64
```

This is computed implicitly as `C // self.n_head` in [model.py:57-59](model.py#L57-L59).

### QKV Projection — [model.py:35](model.py#L35)

```python
c_attn = nn.Linear(n_embd, 3 * n_embd)   # projects to Q, K, V for ALL heads at once
```

The output `(B, T, 2304)` is split into three `(B, T, 768)` tensors, then each is reshaped to `(B, 12, T, 64)` — one slice per head.

### Attention Computation — [model.py:62-71](model.py#L62-L71)

- **Flash Attention** (PyTorch ≥ 2.0): uses `scaled_dot_product_attention`, fused CUDA kernel
- **Fallback**: manual `(Q @ K^T) / sqrt(64)` → causal mask → softmax → `@ V`

After all heads compute, outputs are concatenated back: `(B, 12, T, 64)` → `(B, T, 768)` at [model.py:72](model.py#L72).

---

## Default Config (GPT-2 size) — [train.py:51-55](train.py#L51-L55)

| Parameter | Value | Meaning |
|---|---|---|
| `n_embd` | 768 | Embedding / hidden dimension |
| `n_head` | 12 | Number of attention heads |
| `head_size` | 64 | `n_embd / n_head` (implicit) |
| `block_size` | 1024 | Max sequence length (context window) |
| `n_layer` | 12 | Number of transformer blocks |

---

## GPT Variant Sizes — [model.py:217-221](model.py#L217-L221)

| Model | `n_layer` | `n_head` | `n_embd` | Head size | Params |
|---|---|---|---|---|---|
| gpt2 | 12 | 12 | 768 | 64 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 64 | 350M |
| gpt2-large | 36 | 20 | 1280 | 64 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 64 | 1558M |

> Note: all variants keep `head_size = 64` — scaling is done by adding more heads and layers, not by changing the per-head dimension.

---

## MLP Hidden Dimension — [model.py:82](model.py#L82)

The feedforward block expands to `4 × n_embd = 3072` then projects back to 768. This 4x expansion is standard in transformers.

```python
c_fc   = nn.Linear(n_embd, 4 * n_embd)   # 768 → 3072
c_proj = nn.Linear(4 * n_embd, n_embd)   # 3072 → 768
```
