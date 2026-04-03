# N-gram Speculative Decoding

Speculative decoding is a technique that generates multiple tokens per forward pass, making inference faster without changing the output quality. newsweep implements a variant that uses n-gram patterns from the prompt itself, requiring no separate draft model.

## The problem: sequential token generation

Standard autoregressive generation produces one token per forward pass:

```
Forward pass 1 -> token A
Forward pass 2 -> token B
Forward pass 3 -> token C
...
```

Each forward pass through a 7B model takes ~10-20ms on Apple Silicon. For a 50-token completion, that's 500-1000ms of sequential computation.

## The idea: propose and verify

Instead of generating one token at a time, speculative decoding works in two phases:

1. **Propose**: Cheaply guess what the next K tokens will be
2. **Verify**: Run one forward pass through the main model with all K tokens, checking if the guesses were correct

If the guesses are right, you get K+1 tokens from a single (slightly more expensive) forward pass. If a guess is wrong, you still get at least 1 correct token plus the model's actual prediction.

## Why n-grams work for code completion

In a next-edit prediction task, the prompt contains the **original code** and the model generates an **updated version**. Much of the output is identical to the input -- the model reproduces unchanged lines verbatim.

This means prompt tokens are excellent predictors of output tokens. If the model just generated tokens matching "def main(", the next tokens likely match what follows "def main(" in the prompt.

## How it works

### Step 1: Build the n-gram index

Before generation starts, we build an index from the prompt tokens. For each position in the prompt, we record what token follows each n-gram context:

```python
# Prompt tokens: [10, 20, 30, 40, 50, 20, 30, 60]
# With NGRAM_N=4, we index contexts of size 1, 2, and 3:

# Size-1 contexts (bigrams):
(10,) -> [1]       # token at position 1 follows token 10
(20,) -> [2, 6]    # token at positions 2 and 6 follows token 20
(30,) -> [3, 7]    # etc.
(40,) -> [4]
(50,) -> [5]

# Size-2 contexts (trigrams):
(10, 20) -> [2]
(20, 30) -> [3, 7]
(30, 40) -> [4]
(40, 50) -> [5]
(50, 20) -> [6]

# Size-3 contexts (4-grams):
(10, 20, 30) -> [3]
(20, 30, 40) -> [4]
(30, 40, 50) -> [5]
(40, 50, 20) -> [6]
(50, 20, 30) -> [7]
```

### Step 2: Propose draft tokens

After generating a base token, we look up the last N-1 generated tokens in the index. We try the longest context first (most specific), falling back to shorter ones:

```python
# Just generated: [..., 50, 20, 30]
# Try 3-token context: (50, 20, 30) -> found at position 7
#   -> draft = [token_at_7, token_at_8, token_at_9] (up to NUM_DRAFT_TOKENS)

# If no 3-token match, try 2-token: (20, 30) -> found at positions 3 and 7
#   -> use LAST occurrence (position 7)

# If no 2-token match, try 1-token: (30,) -> found at positions 3 and 7
```

When a pattern appears multiple times, we use the **last** occurrence. This is a heuristic -- in code, later occurrences of a pattern are more likely to be in the relevant context.

### Step 3: Verify in batch

We feed the base token and draft tokens through the main model in a single forward pass:

```
Input to model:  [base_token, draft_0, draft_1, draft_2]
                      |           |         |         |
Model output:    [pred_for_0, pred_for_1, pred_for_2, pred_for_3]
```

- `pred_for_0` is what the model predicts should follow `base_token` -- compare with `draft_0`
- `pred_for_1` is what follows `draft_0` -- compare with `draft_1`
- And so on

We accept consecutive matches from the start:
- If all match: we got 4 tokens from 1 forward pass (4x speedup for this step)
- If `draft_1` doesn't match: we accept `draft_0`, reject the rest, and use `pred_for_1` as the next token
- If `draft_0` doesn't match immediately: we use `pred_for_0` as the next token (no speedup, but no slowdown either)

```
Example - all accepted:
  draft:    [A, B, C]
  verified: [A, B, C]  -> accept all, model also gives next prediction
  result:   4 tokens from 1 pass

Example - partial accept:
  draft:    [A, B, C]
  verified: [A, X, _]  -> accept A, reject B onwards, use X as next token
  result:   3 tokens from 1 pass (A + X + the base token)

Example - none accepted:
  draft:    [A, B, C]
  verified: [Y, _, _]  -> reject all, use Y as next token
  result:   2 tokens from 1 pass (base token + Y)
```

### Step 4: Cache management

After verification, the KV cache must be trimmed to remove entries for rejected tokens. The cache includes entries for all tokens fed to the model (base + all drafts). Rejected token entries are trimmed from the end:

```
Cache after verification:  [base, draft_0, draft_1, draft_2]
If draft_1 rejected:       [base, draft_0]  (trimmed 2 entries)
```

When all drafts are accepted, no trimming is needed -- all cache entries are valid.

If the draft was partially accepted, we also need fresh logits for the next iteration. Since the rejected model prediction was used as the continuation token, we feed it through the model to get logits for the next base token.

## Configuration

| Setting | Default | Effect |
|---|---|---|
| `USE_NGRAM_SPECULATION` | `True` | Enable/disable n-gram speculation |
| `NGRAM_N` | `4` | N-gram size (matches last 3 tokens by default) |
| `NUM_DRAFT_TOKENS` | `3` | Tokens proposed per step |

## Acceptance rates

On typical code edits, n-gram speculation achieves 60-90% acceptance rates. It works best when:
- The completion reproduces existing code (refactoring, small edits)
- The code has repetitive patterns (boilerplate, similar function signatures)

It works less well when:
- The model generates entirely new code
- The completion diverges significantly from the prompt

## Comparison with draft-model speculation

| | N-gram | Draft model |
|---|---|---|
| Extra model | No | Yes (Qwen2.5-0.5B, ~500 MB) |
| Memory | No overhead | ~1 GB |
| Startup time | Instant | Several seconds |
| Best for | Code rewrites | Novel code |
| Acceptance rate | Higher on edits | More consistent |

newsweep defaults to n-gram speculation. The draft model is a fallback available by setting `USE_NGRAM_SPECULATION = False` in `config.py`.
