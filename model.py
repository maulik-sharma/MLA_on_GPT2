"""
GPT-2 with Multi-Head Latent Attention (MLA).

MLA (DeepSeek-V2, https://arxiv.org/abs/2405.04434) replaces the per-head keys and
values of standard multi-head attention with a single low-rank *latent* vector per
token. Keys and values are reconstructed from that latent by up-projections, so
during generation only the latent (plus a small shared RoPE key) needs to be
cached, which makes the KV-cache much smaller.

Everything else in the network follows GPT-2: token embedding, pre-LayerNorm
transformer blocks with a 4x GELU MLP, a final LayerNorm and an LM head tied to the
token embedding. Setting `attn_type="mha"` swaps in standard GPT-2 attention, which
gives a baseline to compare against.

Run `python model.py` to execute a small self-test.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Configuration: change these to scale the model up or down.
# --------------------------------------------------------------------------------------
@dataclass
class GPTConfig:
    # --- general GPT-2 settings ---
    vocab_size: int = 50304      # GPT-2 BPE has 50257 tokens; padded to a multiple of 64 for speed
    block_size: int = 256        # maximum context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True            # bias in Linear/LayerNorm layers (GPT-2 uses True)

    # --- attention type ---
    attn_type: str = "mla"       # "mla" (Multi-Head Latent Attention) or "mha" (standard GPT-2)
    use_abs_pos_emb: bool = False  # learned GPT-2 position embedding (wpe). Always on for "mha".

    # --- MLA settings ---
    q_lora_rank: int = 192       # query compression dimension (0 = no compression of queries)
    kv_lora_rank: int = 128      # KV latent dimension d_c (this is what gets cached)
    qk_nope_head_dim: int = 64   # per-head query/key dims WITHOUT RoPE (from the latent)
    qk_rope_head_dim: int = 32   # per-head query/key dims WITH RoPE (decoupled RoPE)
    v_head_dim: int = 64         # per-head value dims
    rope_theta: float = 10000.0


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square norm, used on the MLA latents (as in DeepSeek-V2)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim, max_len, theta=10000.0):
    """Returns cos, sin tables of shape (max_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)              # (max_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)       # (max_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """x: (..., T, D); cos/sin: (T, D) and broadcast over the leading dims."""
    return (x * cos + rotate_half(x) * sin).to(x.dtype)


def attention(q, k, v, dropout_p):
    """
    q: (B, H, Tq, D), k: (B, H, Tk, D), v: (B, H, Tk, Dv). Causal attention where the
    queries are the LAST Tq positions of the Tk-long sequence (needed with a KV-cache).
    """
    Tq, Tk = q.size(-2), k.size(-2)
    if Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
    if Tq == 1:  # single new token can attend to everything in the cache
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    # general case: causal mask aligned to the bottom-right corner
    mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)


# --------------------------------------------------------------------------------------
# Multi-Head Latent Attention
# --------------------------------------------------------------------------------------
class MultiHeadLatentAttention(nn.Module):
    """
    For an input x (B, T, C):

      queries:  c_q  = RMSNorm(W_DQ x)                    (B, T, q_lora_rank)
                [q_nope ; q_rope] = W_UQ c_q              per head: nope_dim + rope_dim
      keys/vals:[c_kv ; k_rope] = W_DKV x                 (B, T, kv_lora_rank + rope_dim)
                c_kv   = RMSNorm(c_kv)                    <- the compressed latent
                k_nope = W_UK c_kv                        per head: nope_dim
                v      = W_UV c_kv                        per head: v_head_dim
      RoPE on q_rope and k_rope only ("decoupled RoPE"). k_rope is shared by all heads.
      q = [q_nope ; RoPE(q_rope)],  k = [k_nope ; RoPE(k_rope)]
      out = W_O · softmax(q k^T / sqrt(d)) v

    Only (c_kv, k_rope) are cached: (kv_lora_rank + rope_dim) numbers per token per
    layer, compared with 2 * n_head * head_dim for standard MHA.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.nope_dim = cfg.qk_nope_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.dropout = cfg.dropout

        # query path (optionally low-rank)
        if cfg.q_lora_rank > 0:
            self.w_dq = nn.Linear(cfg.n_embd, cfg.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(cfg.q_lora_rank)
            self.w_uq = nn.Linear(cfg.q_lora_rank, cfg.n_head * self.qk_head_dim, bias=False)
        else:
            self.w_q = nn.Linear(cfg.n_embd, cfg.n_head * self.qk_head_dim, bias=False)

        # key/value path: down-projection to the latent + the shared RoPE key
        self.w_dkv = nn.Linear(cfg.n_embd, cfg.kv_lora_rank + self.rope_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.w_uk = nn.Linear(cfg.kv_lora_rank, cfg.n_head * self.nope_dim, bias=False)
        self.w_uv = nn.Linear(cfg.kv_lora_rank, cfg.n_head * self.v_head_dim, bias=False)

        # output projection
        self.c_proj = nn.Linear(cfg.n_head * self.v_head_dim, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True
        self.resid_dropout = nn.Dropout(cfg.dropout)

        cos, sin = precompute_rope(self.rope_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        H = self.n_head
        start = 0 if cache is None else cache[0].size(1)
        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]

        # ---- queries ----
        if hasattr(self, "w_dq"):
            q = self.w_uq(self.q_norm(self.w_dq(x)))
        else:
            q = self.w_q(x)
        q = q.view(B, T, H, self.qk_head_dim).transpose(1, 2)            # (B, H, T, qk_head_dim)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)

        # ---- compressed KV latent + shared RoPE key ----
        c_kv, k_rope = self.w_dkv(x).split([self.kv_lora_rank, self.rope_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)                                       # (B, T, kv_lora_rank)
        k_rope = apply_rope(k_rope, cos, sin)                           # (B, T, rope_dim)

        # append to the cache (only the small latents are stored)
        if cache is not None:
            c_kv = torch.cat([cache[0], c_kv], dim=1)
            k_rope = torch.cat([cache[1], k_rope], dim=1)
        new_cache = (c_kv, k_rope)
        Tk = c_kv.size(1)

        # ---- reconstruct per-head keys and values from the latent ----
        k_nope = self.w_uk(c_kv).view(B, Tk, H, self.nope_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, Tk, H, self.v_head_dim).transpose(1, 2)
        k_rope = k_rope.unsqueeze(1).expand(B, H, Tk, self.rope_dim)   # shared across heads

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        y = attention(q, k, v, self.dropout if self.training else 0.0)  # (B, H, T, v_head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, H * self.v_head_dim)
        return self.resid_dropout(self.c_proj(y)), new_cache


# --------------------------------------------------------------------------------------
# Standard GPT-2 attention (baseline)
# --------------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        y = attention(q, k, v, self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), (k, v)


# --------------------------------------------------------------------------------------
# Transformer block + GPT
# --------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = MultiHeadLatentAttention(cfg) if cfg.attn_type == "mla" else CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, cache=None):
        a, new_cache = self.attn(self.ln_1(x), cache)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.attn_type in ("mla", "mha"), f"unknown attn_type {cfg.attn_type}"
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # MHA has no RoPE, so it needs GPT-2's learned position embedding.
        self.use_wpe = cfg.use_abs_pos_emb or cfg.attn_type == "mha"
        if self.use_wpe:
            self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying, as in GPT-2

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2 init; residual projections scaled by 1/sqrt(2 * n_layer)
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_PROJ", False):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.use_wpe:
                n -= self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, caches=None):
        """
        idx: (B, T) token ids. targets: (B, T) or None.
        caches: per-layer KV-caches from a previous call (used by generate()).
        Returns (logits, loss, new_caches).
        """
        B, T = idx.shape
        start = 0 if caches is None else self._cache_len(caches)
        assert start + T <= self.config.block_size, \
            f"sequence length {start + T} exceeds block_size {self.config.block_size}"

        x = self.wte(idx)
        if self.use_wpe:
            x = x + self.wpe(torch.arange(start, start + T, device=idx.device))
        x = self.drop(x)

        new_caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, None if caches is None else caches[i])
            new_caches.append(c)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # only the last position is needed for sampling
            loss = None
        return logits, loss, new_caches

    def _cache_len(self, caches):
        # MLA cache: (c_kv (B,T,r), k_rope (B,T,d)); MHA cache: (k (B,H,T,d), v)
        return caches[0][0].size(1) if self.config.attn_type == "mla" else caches[0][0].size(2)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """Autoregressive sampling. With use_cache=True only the new token is processed each step."""
        bs = self.config.block_size
        caches = None
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _, _ = self(idx[:, -bs:])
            else:
                if caches is not None and self._cache_len(caches) >= bs:
                    caches = None  # context full: restart the cache on a sliding window
                if caches is None:
                    logits, _, caches = self(idx[:, -(bs // 2 if idx.size(1) >= bs else bs):])
                else:
                    logits, _, caches = self(idx[:, -1:], caches=caches)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    # ---------------- KV-cache size: the main benefit of MLA ----------------
    def kv_cache_elems_per_token(self):
        c = self.config
        if c.attn_type == "mla":
            return c.n_layer * (c.kv_lora_rank + c.qk_rope_head_dim)
        return c.n_layer * 2 * c.n_embd

    def mha_equivalent_kv_cache_elems_per_token(self):
        """Cache size of a standard GPT-2 MHA model with the same n_layer / n_embd."""
        return self.config.n_layer * 2 * self.config.n_embd

    # ---------------- optimizer / checkpoint helpers ----------------
    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]      # weight matrices + embeddings
        no_decay = [p for p in params if p.dim() < 2]    # biases and norm weights
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(GPTConfig(**ckpt["model_config"]))
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        return model.to(device), ckpt


def config_to_dict(cfg: GPTConfig):
    return asdict(cfg)


# --------------------------------------------------------------------------------------
# Self-test: python model.py
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    for attn_type in ("mla", "mha"):
        cfg = GPTConfig(attn_type=attn_type, block_size=64, n_layer=2, n_head=4, n_embd=128,
                        q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=32, qk_rope_head_dim=16,
                        v_head_dim=32)
        m = GPT(cfg).eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 20))

        # 1) shapes + loss
        logits, loss, _ = m(idx, targets=idx)
        assert logits.shape == (2, 20, cfg.vocab_size), logits.shape
        print(f"[{attn_type}] params={m.num_params() / 1e6:.2f}M  initial loss={loss.item():.3f} "
              f"(ln(vocab)={math.log(cfg.vocab_size):.3f})")

        # 2) causality: changing a future token must not change earlier logits
        idx2 = idx.clone()
        idx2[:, 15] = (idx2[:, 15] + 1) % cfg.vocab_size
        l2, _, _ = m(idx2, targets=idx2)
        assert torch.allclose(logits[:, :15], l2[:, :15], atol=1e-5), "causality violated"

        # 3) KV-cache: prefill + incremental decoding == full forward pass
        _, _, caches = m(idx[:, :12])
        outs = []
        for t in range(12, 20):
            lt, _, caches = m(idx[:, t:t + 1], caches=caches)
            outs.append(lt[:, -1])
        inc = torch.stack(outs, dim=1)
        assert torch.allclose(inc, logits[:, 12:20], atol=1e-4), (inc - logits[:, 12:20]).abs().max()

        # 4) multi-token chunk with a cache (bottom-right causal mask)
        _, _, caches = m(idx[:, :10])
        lc, _, _ = m(idx[:, 10:20], targets=idx[:, 10:20], caches=caches)
        assert torch.allclose(lc, logits[:, 10:20], atol=1e-4)

        # 5) generation runs past block_size
        out = m.generate(idx[:, :5], max_new_tokens=80, top_k=10)
        assert out.shape == (2, 85)

        print(f"[{attn_type}] KV-cache elements/token: {m.kv_cache_elems_per_token()} "
              f"(MHA equivalent: {m.mha_equivalent_kv_cache_elems_per_token()})  -- all tests passed")
