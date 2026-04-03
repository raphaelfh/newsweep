# Token Healing

Token healing fixes a subtle problem with autocomplete: when users type partial words, tokenization produces sequences the model has never seen in training, leading to low-quality completions.

## The problem

Suppose the user types `Nod` (intending `Node`). The tokenizer might encode this as:

```
"Nod" -> [token_id_for_"N", token_id_for_"od"]
```

But in training data, the model always saw `Node` tokenized as:

```
"Node" -> [token_id_for_"Node"]
```

The model has never seen the sequence `["N", "od"]` followed by the letter `e`. It doesn't know what to do -- so it generates garbage or unrelated completions.

This is an **unstable tokenization boundary**: the same text can tokenize differently depending on what comes after it.

## The solution

Token healing works in three steps:

### 1. Detect the unstable boundary

Walk backward from the end of the token list. For each position, decode the tail tokens to text, then re-encode. If re-encoding produces different token IDs, that's an unstable boundary:

```
Tokens: [..., tok_A, tok_B, tok_C]

Check last 1 token:
  decode([tok_C]) = "od"
  encode("od")    = [tok_C]  -> same, boundary is stable here

Check last 2 tokens:
  decode([tok_B, tok_C]) = "Nod"
  encode("Nod")          = [tok_X]  -> different! Unstable boundary.

Roll back 1 token (the unstable one):
  trimmed_tokens = [..., tok_A, tok_B]
  healing_prefix = "od"
```

The healing prefix (`"od"`) is the text that was removed. The model must now regenerate it.

### 2. Constrain generation

A **logits processor** is created that forces the first few generated tokens to match the healing prefix. On each generation step:

1. Look up which tokens in the vocabulary start with (or are a prefix of) the remaining healing prefix
2. Set all other token logits to `-inf` (mask them out)
3. The sampler can only pick tokens consistent with the prefix

```
Healing prefix: "od"

Step 1:
  Allowed tokens: "ode", "od", "o" (and other tokens starting with "od" or that "od" starts with)
  If model picks "o": remaining prefix = "d"
  If model picks "ode": remaining prefix = "" (fully consumed)

Step 2 (if prefix remains):
  Allowed tokens: tokens starting with "d" or that "d" starts with
  Model picks "de_": remaining prefix = "" (consumed, "de_" starts with "d")

After prefix consumed: processor becomes no-op, model generates freely.
```

### 3. Strip the healing prefix

The generated text starts with the healing prefix (e.g., `"ode_function..."`), but the IDE already has `"od"` on screen. The server strips the healing prefix from the completion before returning it.

## The vocabulary trie

To quickly find which tokens match a given prefix, newsweep builds a **trie** (prefix tree) from the tokenizer's vocabulary at startup:

```
root
 ├─ N
 │  ├─ o
 │  │  ├─ d  -> [tok_id_"Nod"]
 │  │  ├─ de -> [tok_id_"Node"]
 │  │  └─ ...
 │  └─ ...
 ├─ o
 │  ├─ d  -> [tok_id_"od"]
 │  └─ ...
 └─ ...
```

Two types of lookups:

- **`prefix_search("od")`**: Find all tokens whose text starts with `"od"` -- these tokens *extend* the prefix (e.g., `"ode"`, `"ods"`, `"od"`)
- **`continuation_search("od")`**: Find all tokens that either extend the prefix OR are consumed by it (e.g., `"o"` is a valid first step because `"od"` starts with `"o"`)

The healing processor uses `continuation_search` to allow both multi-character matches (`"ode"` consumes the whole prefix in one token) and incremental matches (`"o"` consumes part of it).

## Performance optimization: prefix cache

Some single characters (like space) are prefixes of thousands of tokens. Computing the allowed set each time would be slow. At startup, `warmup_prefix_cache()` pre-computes allowed token sets for high-frequency single-character prefixes (those appearing as first characters in 1000+ vocabulary entries).

This brings healing overhead to sub-1ms for the common case.

## Configuration

| Setting | Default | Description |
|---|---|---|
| `ENABLE_TOKEN_HEALING` | `True` | Enable/disable healing |
| `MAX_HEALING_TOKENS` | `3` | Max tokens to roll back when detecting boundary |

## When healing matters most

Token healing makes the biggest difference when:
- Users type partial identifiers (common with autocomplete triggers)
- Variable names or keywords are split across unusual token boundaries
- The model would otherwise need to "guess" what comes after an incomplete token

It has no effect when the tokenization boundary is already stable (which is most of the time). In those cases, the processor detects stability and skips healing entirely.
