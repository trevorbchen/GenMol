"""
QED Surrogate: a small 1D UNet that predicts QED from clean token sequences.

Learns R_phi(x0_tokens) ≈ QED(decode(x0_tokens)), providing a differentiable
reward signal for DPS-style guidance during MDLM denoising.

Architecture adapted from TTT's MeasurementPredictor UNet:
  - Pre-activation ResBlocks: GroupNorm -> SiLU -> Conv1d (x2) + skip
  - Zero-initialized second conv (network starts predicting zero)
  - Encoder-decoder with skip connections
  - Self-attention at bottleneck
  - Global avg pool -> scalar output head

Operates on 1D token sequences (Conv1d) instead of 2D images (Conv2d).
No sigma conditioning since input is always clean x0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers (from TTT)
# ---------------------------------------------------------------------------

def zero_module(module):
    """Zero-initialize all parameters of a module (TTT convention)."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ---------------------------------------------------------------------------
# 1D UNet components (adapted from TTT's MeasurementPredictor)
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """Pre-activation ResBlock for 1D sequences.

    GroupNorm -> SiLU -> Conv1d -> GroupNorm -> SiLU -> zero_Conv1d + skip.
    Adapted from TTT's ResBlock (2D) but without FiLM sigma conditioning
    since our input is always clean x0.
    """

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1))
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv1d(in_ch, out_ch, kernel_size=1))

    def forward(self, x):
        """x: [B, C, L] -> [B, C_out, L]"""
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return self.skip(x) + h


class SelfAttention1d(nn.Module):
    """Multi-head self-attention on 1D sequences.

    Adapted from TTT's SelfAttention (2D). Flattening is trivial for 1D.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.out_proj = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x):
        """x: [B, C, L] -> [B, C, L]"""
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv.unbind(1)  # each [B, H, D, L]

        scale = self.head_dim ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", q * scale, k)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        h = h.reshape(B, C, L)

        return x + self.out_proj(h)


# ---------------------------------------------------------------------------
# QED Surrogate (1D UNet)
# ---------------------------------------------------------------------------

class QEDSurrogate(nn.Module):
    """
    Small 1D UNet that predicts QED from clean token sequences.

    Architecture (adapted from TTT's MeasurementPredictor):
      - Token embedding: Embedding(vocab_size, embed_dim)
      - Input conv: Conv1d(embed_dim, C, 1)
      - Encoder: N levels, each with ResBlock1d + stride-2 downsample
      - Bottleneck: ResBlock1d + SelfAttention1d + ResBlock1d
      - Decoder: mirrors encoder with skip connections
      - Output: GroupNorm -> SiLU -> global avg pool -> MLP -> scalar

    Channel progression with default channel_mult=(1, 2, 4, 4):
      Encoder: [C, 2C, 4C, 4C]  (C=64 -> [64, 128, 256, 256])

    Input:  x0 token_ids [B, L] (clean, fully denoised sequences)
    Output: predicted QED [B] (scalar in ~[0, 1])

    For guided sampling, forward_soft() accepts soft token distributions
    [B, L, V] and produces differentiable QED predictions.
    """

    def __init__(
        self,
        vocab_size=1882,
        max_seq_len=256,
        embed_dim=128,
        base_channels=64,
        channel_mult=(1, 2, 4, 4),
        attn_heads=4,
        num_res_blocks=1,
        dropout=0.1,
        pad_token_id=3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.base_channels = base_channels
        self.channel_mult = list(channel_mult)
        self.pad_token_id = pad_token_id
        self.num_res_blocks = num_res_blocks

        C = base_channels
        ch_list = [C * m for m in channel_mult]
        num_levels = len(channel_mult)
        bot_ch = ch_list[-1]

        # -- Token embedding ---------------------------------------------------
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        # -- Input projection: embed_dim -> base_channels ----------------------
        self.input_conv = nn.Conv1d(embed_dim, C, kernel_size=1)

        # -- Encoder -----------------------------------------------------------
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = C
        for i, cur_ch in enumerate(ch_list):
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                level_blocks.append(ResBlock1d(
                    prev_ch if j == 0 else cur_ch, cur_ch, dropout=dropout))
            self.enc_blocks.append(level_blocks)
            self.downsamples.append(
                nn.Conv1d(cur_ch, cur_ch, kernel_size=3, stride=2, padding=1))
            prev_ch = cur_ch

        # -- Bottleneck: ResBlock + SelfAttention + ResBlock -------------------
        self.bot_res1 = ResBlock1d(bot_ch, bot_ch, dropout=dropout)
        self.bot_attn = SelfAttention1d(bot_ch, num_heads=attn_heads)
        self.bot_res2 = ResBlock1d(bot_ch, bot_ch, dropout=dropout)

        # -- Decoder -----------------------------------------------------------
        self.upsample_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        dec_prev_ch = bot_ch
        for i in reversed(range(num_levels)):
            skip_ch = ch_list[i]
            dec_out_ch = ch_list[i - 1] if i > 0 else C
            self.upsample_layers.append(
                nn.Upsample(scale_factor=2, mode="nearest"))
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = (dec_prev_ch + skip_ch) if j == 0 else dec_out_ch
                level_blocks.append(ResBlock1d(in_ch, dec_out_ch, dropout=dropout))
            self.dec_blocks.append(level_blocks)
            dec_prev_ch = dec_out_ch

        # -- Output head: pool -> scalar (zero-init, TTT convention) -----------
        self.out_norm = nn.GroupNorm(min(32, C), C)
        self.out_head = nn.Sequential(
            nn.Linear(C, C),
            nn.SiLU(),
            zero_module(nn.Linear(C, 1)),
        )

    def _encode(self, x):
        """Shared encoder-decoder backbone.

        Args:
            x: [B, embed_dim, L] token embeddings (channel-first)
        Returns:
            [B, C] pooled features
        """
        # Input projection
        h = self.input_conv(x)  # [B, C, L]

        # Encoder
        skips = []
        for level_blocks, down in zip(self.enc_blocks, self.downsamples):
            for block in level_blocks:
                h = block(h)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bot_res1(h)
        h = self.bot_attn(h)
        h = self.bot_res2(h)

        # Decoder
        for level_blocks, up in zip(self.dec_blocks, self.upsample_layers):
            skip = skips.pop()
            h = up(h)
            # Match sequence lengths (may differ by 1 due to stride-2)
            if h.shape[-1] != skip.shape[-1]:
                h = h[..., :skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            for block in level_blocks:
                h = block(h)

        # Output: norm -> act -> global avg pool -> scalar head
        h = F.silu(self.out_norm(h))  # [B, C, L]
        h = h.mean(dim=-1)            # [B, C]
        return self.out_head(h).squeeze(-1)  # [B]

    def forward(self, token_ids, attention_mask=None):
        """
        Standard forward: integer token IDs -> QED prediction.

        Args:
            token_ids:      [B, L] integer token IDs
            attention_mask: [B, L] (unused, kept for interface compat)
        Returns:
            [B] predicted QED scores
        """
        x = self.token_emb(token_ids)       # [B, L, D]
        x = x.transpose(1, 2)               # [B, D, L] channel-first for Conv1d
        return self._encode(x)

    def forward_soft(self, token_probs, attention_mask=None):
        """
        Differentiable forward: soft token distributions -> QED prediction.

        Used during guided sampling. Gradients flow through the soft
        embedding lookup (matrix multiply).

        Args:
            token_probs:    [B, L, V] softmax probabilities over vocab
            attention_mask: [B, L] (unused, kept for interface compat)
        Returns:
            [B] predicted QED scores (differentiable w.r.t. token_probs)
        """
        x = torch.matmul(token_probs, self.token_emb.weight)  # [B, L, D]
        x = x.transpose(1, 2)                                  # [B, D, L]
        return self._encode(x)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_surrogate(model, path, **metadata):
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": model.vocab_size,
            "max_seq_len": model.max_seq_len,
            "embed_dim": model.embed_dim,
            "base_channels": model.base_channels,
            "channel_mult": model.channel_mult,
            "num_res_blocks": model.num_res_blocks,
            "pad_token_id": model.pad_token_id,
        },
        **metadata,
    }, path)


def load_surrogate(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = QEDSurrogate(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)
