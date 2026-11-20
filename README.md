# Multi-Head Latent Attention on GPT-2

This project implements Multi-Head Latent Attention (MLA), the attention mechanism introduced in DeepSeek-V2, inside a GPT-2 style decoder-only language model. The model is trained from scratch on the [`nampdn-ai/tiny-webtext`](https://huggingface.co/datasets/nampdn-ai/tiny-webtext) dataset.

The code base is deliberately small and self-contained. Apart from the attention layer, the network follows GPT-2. A standard multi-head attention (MHA) baseline is included so the two mechanisms can be compared directly.

## Contents

- [Background](#background)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Dataset access](#dataset-access)
- [Usage](#usage)
- [Command-line reference](#command-line-reference)
- [Scaling the model](#scaling-the-model)
- [Outputs](#outputs)
- [Testing](#testing)
- [References](#references)

## Background

In standard multi-head attention, every layer stores a key and a value vector for every head and every token that has already been processed. This is the KV-cache. For a model with `n_layer` layers and hidden size `n_embd`, the cache grows by `2 * n_layer * n_embd` values per generated token. At long context lengths it becomes the main memory cost of inference.

MLA reduces this cost with a low-rank joint compression of keys and values. Each token is projected to a single latent vector `c_kv` of dimension `kv_lora_rank`, which is much smaller than `2 * n_embd`. The per-head keys and values are reconstructed from the latent with learned up-projections. Only the latent needs to be cached.

Rotary position embeddings (RoPE) cannot be applied directly to keys reconstructed from a cached latent. MLA therefore uses decoupled RoPE: a small extra key component of dimension `qk_rope_head_dim` carries the positional information. It is computed directly from the input, shared across all heads, and cached alongside the latent.

## Architecture

### Overall model

| Component | Description |
|---|---|
| Token embedding | `vocab_size x n_embd`, weights tied with the output head |
| Position information | Decoupled RoPE inside MLA. The learned GPT-2 position embedding is optional for MLA and always used for the MHA baseline. |
| Transformer block (x `n_layer`) | Pre-LayerNorm, attention, residual, pre-LayerNorm, MLP (4x expansion, GELU), residual |
| Final layer | LayerNorm followed by the language-model head |
| Initialisation | GPT-2 scheme: normal(0, 0.02), with residual projections scaled by `1 / sqrt(2 * n_layer)` |

### Multi-Head Latent Attention

For an input `x` of shape `(B, T, n_embd)`:

```
Query path
    c_q              = RMSNorm(W_DQ x)                 (B, T, q_lora_rank)
    [q_nope ; q_rope] = W_UQ c_q                        per head: qk_nope_head_dim + qk_rope_head_dim

Key / value path
    [c_kv ; k_rope]  = W_DKV x                          (B, T, kv_lora_rank + qk_rope_head_dim)
    c_kv             = RMSNorm(c_kv)                    compressed latent (cached)
    k_nope           = W_UK c_kv                        per head: qk_nope_head_dim
    v                = W_UV c_kv                        per head: v_head_dim

Positional encoding
    q_rope, k_rope   = RoPE(q_rope), RoPE(k_rope)       k_rope is shared by all heads (cached)

Attention
    q   = [q_nope ; q_rope]
    k   = [k_nope ; k_rope]
    out = W_O softmax(q k^T / sqrt(d)) v
```

If `q_lora_rank` is set to `0`, the query path uses a single full-rank projection instead of the down/up pair.

The attention computation uses `torch.nn.functional.scaled_dot_product_attention`, which selects FlashAttention or memory-efficient kernels when they are available.

### KV-cache comparison

| Attention | Cached values per token per layer | Default configuration (6 layers) |
|---|---|---|
| MHA | `2 * n_embd` | 4,608 |
| MLA | `kv_lora_rank + qk_rope_head_dim` | 960 |

With the default settings, MLA reduces the KV-cache by a factor of about 4.8. Both `train.py` and `generate.py` print this comparison for the loaded configuration.

## Project structure

```
.
├── model.py           Model definition: configuration, MLA, MHA baseline, GPT, generation, self-test
├── train.py           Dataset preparation, training loop, evaluation, checkpointing
├── generate.py        Text generation from a saved checkpoint
├── requirements.txt   Python dependencies
├── data/              Tokenised dataset (created by --prepare)
└── out/               Checkpoints and training log (created by training)
```

## Installation

Requires Python 3.10 or later. A CUDA-capable GPU is recommended. The code also runs on Apple Silicon (MPS) and CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the default `torch` wheel does not match your CUDA version, install PyTorch first by following the instructions at https://pytorch.org.

## Dataset access

`nampdn-ai/tiny-webtext` is a gated dataset on the Hugging Face Hub. Before the first download:

1. Open the dataset page and accept its terms of use.
2. Authenticate on the machine:

```bash
hf auth login
```

The dataset has the following text fields:

| Column | Content |
|---|---|
| `human` | Raw web text. This is the default training text. |
| `bot` | Machine-generated commentary on the corresponding web text |

Choose the column with `--text_column`. Several columns can be combined with a comma, for example `--text_column human,bot`. The values from each row are joined with a blank line.

## Usage

### 1. Prepare the data

```bash
python train.py --prepare                      # full dataset
python train.py --prepare --max_docs 20000     # subset for quick experiments
```

Each document is tokenised with the GPT-2 BPE tokenizer (`tiktoken`), and an end-of-text token is appended after it. All tokens are written to `data/train.bin` and `data/val.bin` as flat `uint16` arrays. During training these files are read through `numpy.memmap`, so the dataset never has to fit in RAM.

If both files already exist, preparation is skipped. Delete them to rebuild the data, for example after changing `--max_docs` or `--text_column`.

### 2. Train

```bash
python train.py                                    # default small MLA model
python train.py --compile                          # same, with torch.compile
python train.py --attn_type mha --out_dir out_mha  # standard attention baseline
python train.py --resume                           # continue from out/ckpt_last.pt
```

On each run the script reports the device, precision, parameter count, KV-cache size and tokens per iteration. A progress bar shows the current loss, learning rate, best validation loss and throughput. Every `eval_interval` steps the script estimates the training and validation loss, writes a checkpoint and prints a short generated sample.

Training stops early when the validation loss has not improved by at least `min_delta` for `patience` consecutive evaluations. `ckpt_best.pt` always holds the model with the lowest validation loss. Loss plots are written to `plots/` when training ends.

### 3. Generate text

```bash
python generate.py --ckpt out/ckpt_best.pt --prompt "The history of the internet" --max_new_tokens 200
```

### 4. Comparing MLA and MHA

Train both variants with identical training settings and separate output directories, then compare the `log.csv` files:

```bash
python train.py --attn_type mla --out_dir out_mla
python train.py --attn_type mha --out_dir out_mha
```

## Command-line reference

`train.py` exposes every field of `GPTConfig` (model) and `TrainConfig` (training) as a command-line option. Boolean options can be given as a bare flag (`--compile`) or with an explicit value (`--compile false`).

### train.py: modes

| Option | Description |
|---|---|
| `--prepare` | Download and tokenise the dataset, then exit |
| `--resume` | Resume training from `<out_dir>/ckpt_last.pt`, restoring the model, optimizer, iteration counter and best validation loss. The model configuration is taken from the checkpoint. |

### train.py: model options

| Option | Default | Description |
|---|---|---|
| `--attn_type` | `mla` | Attention mechanism: `mla` or `mha` |
| `--vocab_size` | `50304` | Vocabulary size (GPT-2's 50,257 tokens, padded to a multiple of 64) |
| `--block_size` | `256` | Maximum context length |
| `--n_layer` | `6` | Number of transformer blocks |
| `--n_head` | `6` | Number of attention heads |
| `--n_embd` | `384` | Hidden size |
| `--dropout` | `0.0` | Dropout probability |
| `--bias` | `true` | Use bias terms in linear and LayerNorm layers |
| `--use_abs_pos_emb` | `false` | Add GPT-2's learned position embedding in MLA models. MHA always uses it. |
| `--q_lora_rank` | `192` | Query compression rank. `0` disables query compression. |
| `--kv_lora_rank` | `128` | Dimension of the cached key/value latent |
| `--qk_nope_head_dim` | `64` | Per-head query/key dimension without RoPE |
| `--qk_rope_head_dim` | `32` | Per-head query/key dimension with decoupled RoPE |
| `--v_head_dim` | `64` | Per-head value dimension |
| `--rope_theta` | `10000.0` | RoPE base frequency |

### train.py: data options

| Option | Default | Description |
|---|---|---|
| `--dataset` | `nampdn-ai/tiny-webtext` | Hugging Face dataset identifier |
| `--data_dir` | `data` | Directory for the tokenised `.bin` files |
| `--text_column` | `human` | Text column(s), comma-separated. If none are found, the first string column is used. |
| `--max_docs` | `0` | Limit the number of training documents (`0` = all) |
| `--val_frac` | `0.005` | Fraction held out for validation if the dataset has no validation split. When `--max_docs` is set, it also determines how many validation documents are used. |
| `--num_proc` | half of CPU cores | Worker processes for tokenisation |

### train.py: optimisation options

| Option | Default | Description |
|---|---|---|
| `--batch_size` | `32` | Sequences per micro-batch |
| `--grad_accum_steps` | `1` | Micro-batches per optimizer step |
| `--max_iters` | `5000` | Total optimizer steps |
| `--learning_rate` | `6e-4` | Peak learning rate |
| `--min_lr` | `6e-5` | Final learning rate after cosine decay |
| `--warmup_iters` | `200` | Linear warmup steps |
| `--weight_decay` | `0.1` | AdamW weight decay (applied to matrices and embeddings only) |
| `--beta1` | `0.9` | AdamW beta1 |
| `--beta2` | `0.95` | AdamW beta2 |
| `--grad_clip` | `1.0` | Gradient norm clipping threshold (`0` disables clipping) |

The effective batch size in tokens is `batch_size * grad_accum_steps * block_size`.

### train.py: early stopping options

| Option | Default | Description |
|---|---|---|
| `--patience` | `5` | Stop after this many consecutive evaluations without improvement (`0` disables early stopping) |
| `--min_delta` | `0.001` | Minimum decrease in validation loss that counts as an improvement |

### train.py: logging and checkpointing options

| Option | Default | Description |
|---|---|---|
| `--out_dir` | `out` | Directory for checkpoints and the training log |
| `--eval_interval` | `250` | Steps between evaluations and checkpoints |
| `--eval_iters` | `50` | Batches used to estimate each loss |
| `--log_interval` | `10` | Steps between training-loss log lines |
| `--sample_tokens` | `64` | Length of the sample generated at each evaluation (`0` disables sampling) |

### train.py: system options

| Option | Default | Description |
|---|---|---|
| `--device` | `auto` | `auto`, `cuda`, `cuda:N`, `mps` or `cpu`. `auto` selects CUDA, then MPS, then CPU. |
| `--dtype` | `auto` | `auto`, `bfloat16`, `float16` or `float32`. `auto` uses bfloat16 on GPUs that support it, otherwise float16 with gradient scaling on CUDA, and float32 elsewhere. |
| `--compile` | `false` | Compile the model with `torch.compile` |
| `--seed` | `1337` | Random seed |

### generate.py

| Option | Default | Description |
|---|---|---|
| `--ckpt` | `out/ckpt_best.pt` | Checkpoint to load |
| `--prompt` | empty | Prompt text. If empty, generation starts from an end-of-text token. |
| `--num_samples` | `1` | Number of independent samples |
| `--max_new_tokens` | `200` | Tokens to generate per sample |
| `--temperature` | `0.8` | Sampling temperature |
| `--top_k` | `50` | Restrict sampling to the k most likely tokens |
| `--no_cache` | off | Disable the KV-cache and recompute the full context each step |
| `--device` | `cuda` if available | Device to run on |
| `--seed` | none | Random seed for reproducible samples |

During generation, the model keeps the cached latents and processes only the newest token at each step. When the cache reaches `block_size`, it is rebuilt from the most recent `block_size / 2` tokens, and generation continues on this sliding window.

## Scaling the model

The default configuration has about 10M non-embedding parameters (about 29M in total). It is meant for fast iteration on a single consumer GPU. Larger models are configured entirely from the command line:

| Configuration | Suggested options |
|---|---|
| Small (default) | none |
| Medium | `--n_layer 8 --n_head 8 --n_embd 512 --kv_lora_rank 192 --q_lora_rank 256 --block_size 512` |
| GPT-2 small scale | `--n_layer 12 --n_head 12 --n_embd 768 --kv_lora_rank 256 --q_lora_rank 384 --block_size 1024` |

When the model grows, lower `--batch_size` and raise `--grad_accum_steps` to keep the same effective batch size within GPU memory. The per-head dimensions (`qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`) are independent of `n_embd` and can usually stay at their defaults.

## Outputs

Training writes the following to `--out_dir`:

| File | Content |
|---|---|
| `ckpt_last.pt` | Most recent checkpoint, written at every evaluation |
| `ckpt_best.pt` | Checkpoint with the lowest validation loss so far |
| `log.csv` | Columns `iter`, `train_loss`, `val_loss`, `lr`, one row per evaluation |

Each checkpoint contains the model weights, the optimizer state, both configurations, the iteration number and the best validation loss. To load a model programmatically:

```python
from model import GPT

model, checkpoint = GPT.from_checkpoint("out/ckpt_best.pt", device="cuda")
```

## Testing

`model.py` contains a self-test that runs on CPU in a few seconds:

```bash
python model.py
```

It checks the following for both the MLA and MHA variants:

- Output shapes and initial loss
- Causality: changing a future token does not affect earlier outputs
- Incremental decoding with the KV-cache matches a full forward pass
- Processing a multi-token chunk on top of an existing cache matches a full forward pass
- Generation beyond `block_size`

## References

- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. 2024. arXiv:2405.04434.
- Radford et al. *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019.
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021. arXiv:2104.09864.
- Karpathy. *nanoGPT*. https://github.com/karpathy/nanoGPT. The memory-mapped data pipeline follows the approach used there.
