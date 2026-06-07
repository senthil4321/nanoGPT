# Training Logic Flow — nanoGPT

## 1. The Core Objective: Next-Token Prediction

The model is trained to predict the next token given all previous tokens. In `get_batch()` ([train.py:116-131](train.py#L116-L131)):

- `x` = tokens at positions `[0, 1, ..., block_size-1]`
- `y` = tokens at positions `[1, 2, ..., block_size]` (shifted by 1)

Every position in a sequence is a training example simultaneously — one forward pass trains on `block_size` predictions at once.

---

## 2. The Training Loop (Forward → Backward → Update)

Each iteration does three things ([train.py:292-314](train.py#L292-L314)):

1. **Forward pass**: `model(X, Y)` computes predictions (logits) and cross-entropy loss
2. **Backward pass**: `.backward()` computes gradients via backpropagation
3. **Optimizer step**: AdamW updates the weights using those gradients, then `zero_grad()` clears them

---

## 3. Gradient Accumulation

Because GPUs have limited memory, you can't always use a large batch. The trick ([train.py:292-305](train.py#L292-L305)):

- Run `gradient_accumulation_steps` micro-batches, accumulating gradients each time (no `zero_grad` between them)
- Divide loss by the number of steps so the effective gradient is as if you ran one big batch
- Only then do a single optimizer step

This simulates a larger batch without the memory cost.

---

## 4. Learning Rate Schedule: Cosine with Warmup

Defined in `get_lr()` ([train.py:231-242](train.py#L231-L242)), it has three phases:

- **Warmup** (first ~2000 steps): LR rises linearly from 0 to max — avoids instability at the start
- **Cosine decay**: LR smoothly decreases following a cosine curve
- **Floor**: LR stays at `min_lr` (1/10th of max) after decay is done

---

## 5. Mixed Precision Training

Instead of 32-bit floats everywhere, computations run in `bfloat16` or `float16` ([train.py:112](train.py#L112)):

- **Faster** and uses **less GPU memory**
- A `GradScaler` prevents gradient underflow in `float16` by scaling loss up before backward, then unscaling before the optimizer step

---

## 6. Gradient Clipping

Before the optimizer step ([train.py:308-309](train.py#L308-L309)):

```python
clip_grad_norm_(model.parameters(), grad_clip=1.0)
```

If gradients get very large (exploding gradients), this rescales them to have a maximum norm of 1.0, preventing catastrophic weight updates.

---

## 7. Checkpointing & Evaluation

Every `eval_interval` steps ([train.py:263-286](train.py#L263-L286)):

- The model switches to `.eval()` mode, runs `eval_iters` batches on both train/val sets, averages the loss
- If val loss improved (or `always_save_checkpoint=True`), saves `ckpt.pt` containing model weights, optimizer state, and iteration count — so training can be **resumed exactly**

---

## 8. Distributed Data Parallel (DDP)

When running on multiple GPUs, each GPU holds a full copy of the model and processes a different batch. Gradients are averaged across all GPUs before the weight update. The key optimization ([train.py:298](train.py#L298)): gradient sync is skipped on all micro-steps except the last, avoiding expensive communication on every accumulation step.

---

**The flow in one sentence:** sample random token windows → predict next tokens → measure prediction error (cross-entropy loss) → backpropagate error → update weights to minimize that error → repeat 600,000 times.
