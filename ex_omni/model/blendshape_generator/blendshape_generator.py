import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange, repeat
from torch import einsum
from .blendshape_utils import BlendShapeLoss, units_to_render_frames
from ex_omni.constants import IGNORE_INDEX


class PeriodicRotaryPositionalEmbedding(nn.Module):
    """
    Periodic RoPE
    """

    def __init__(
        self,
        dim: int,
        period: int = 25,
        base: int = 10000,
        scaling_factor: float = 1.0,
        cache_max_len: int = 10000,
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even dimension"

        self.dim = dim
        self.period = period
        self.base = base
        self.scaling_factor = scaling_factor
        self.cache_max_len = cache_max_len
        self._warned_dynamic = False

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        self._build_cache(cache_max_len)


    def _build_cache(self, max_len: int):
        positions = torch.arange(max_len, dtype=torch.float32)

        periodic_pos = (positions % self.period) / self.scaling_factor  # [T]

        freqs = torch.outer(periodic_pos, self.inv_freq)  # [T, D/2]

        emb = freqs.repeat_interleave(2, dim=-1)  # [T, D]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _compute_dynamic(self, seq_len: int, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        periodic_pos = (positions % self.period) / self.scaling_factor

        freqs = torch.outer(periodic_pos, self.inv_freq.to(device))  # [T, D/2]
        emb = freqs.repeat_interleave(2, dim=-1)                      # [T, D]

        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        assert x.shape[-1] == self.dim, \
            f"Last dim of x ({x.shape[-1]}) must equal dim ({self.dim})"

        seq_len = x.shape[seq_dim]
        device, dtype = x.device, x.dtype

        if self.cos_cached.device != device:
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)

        if seq_len <= self.cache_max_len:
            cos_base = self.cos_cached[:seq_len].to(dtype)
            sin_base = self.sin_cached[:seq_len].to(dtype)
        else:
            cos_base, sin_base = self._compute_dynamic(seq_len, device, dtype)
            if not self._warned_dynamic:
                print(f"⚠️ 序列长度 {seq_len} 超出缓存 {self.cache_max_len}，使用动态计算。")
                self._warned_dynamic = True

        shape = [1] * x.dim()
        shape[seq_dim] = seq_len
        shape[-1] = self.dim

        cos = cos_base.view(*shape)
        sin = sin_base.view(*shape)

        return self._apply_rotary_emb(x, cos, sin)


    @staticmethod
    def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        cos_theta = cos[..., 0::2]
        sin_theta = sin[..., 0::2]

        rx1 = x1 * cos_theta - x2 * sin_theta
        rx2 = x1 * sin_theta + x2 * cos_theta

        return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class RoPEWrapper(nn.Module):
    def __init__(self, transformer_encoder, rope_module):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.rope = rope_module
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        src_with_rope = self.rope(src)
        return self.transformer_encoder(
            src_with_rope, 
            mask=mask, 
            src_key_padding_mask=src_key_padding_mask
        )


# ========== 与 Speech Generator 共享的 TQGF 模块 ==========
def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class TQGFGatedCrossAttention(nn.Module):
    def __init__(self, dim=512, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        
        self.norm_text = nn.LayerNorm(dim)
        self.norm_query = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        self.gate_proj = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, text_reps, query_reps, text_mask=None):
        text_reps_normed = self.norm_text(text_reps)
        query_reps_normed = self.norm_query(query_reps)
        
        h, d = self.heads, self.dim_head
        
        q = rearrange(self.to_q(query_reps_normed), "b n (h d) -> b h n d", h=h, d=d)
        k, v = self.to_kv(text_reps_normed).chunk(2, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=h, d=d)
        v = rearrange(v, "b n (h d) -> b h n d", h=h, d=d)
        
        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        
        if text_mask is not None:
            mask = rearrange(text_mask, "b n -> b 1 1 n")
            sim = sim.masked_fill(~mask, float('-inf'))
        
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        
        gate_scores = self.gate_proj(query_reps_normed)
        gate_scores = rearrange(gate_scores, "b n (h d) -> b h n d", h=h, d=d)
        gate_scores = torch.sigmoid(gate_scores)
        out = out * gate_scores
        
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TQGF(nn.Module):
    def __init__(self, dim, depth=2, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    TQGFGatedCrossAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ])
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_reps, query_reps, text_mask=None):
        for attn, ff in self.layers:
            query_reps = attn(text_reps, query_reps, text_mask=text_mask) + query_reps
            query_reps = ff(query_reps) + query_reps
        return self.norm(query_reps)


class BlendshapeGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.speech_gen_projection = nn.Linear(config.speech_gen_hidden_size, config.bs_decoder_dimension)
        self.speech_token_embedding = nn.Embedding(config.speech_vocab_size, config.speech_embed_dim)

        self.speech_token_projection = nn.Linear(config.speech_embed_dim, config.bs_decoder_dimension)
        
        tqgf_depth = getattr(config, 'bs_tqgf_depth', None) or 2
        tqgf_dim_head = getattr(config, 'bs_tqgf_dim_head', None) or 64
        tqgf_heads = getattr(config, 'bs_tqgf_heads', None) or 8
        self.tqgf = TQGF(
            dim=config.bs_decoder_dimension,
            depth=tqgf_depth,
            dim_head=tqgf_dim_head,
            heads=tqgf_heads,
            ff_mult=4
        )
        
        rope_period = getattr(config, 'period', 30)
        rope_cache_max_len = getattr(config, 'rope_cache_max_len', 10000)
        
        self.rope = PeriodicRotaryPositionalEmbedding(
            dim=config.bs_decoder_dimension,
            period = rope_period,
            base = 10000,
            scaling_factor = 1.0,
            cache_max_len = rope_cache_max_len
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.bs_decoder_dimension,
            nhead=config.nhead,
            dim_feedforward=config.bs_decoder_dimension * 2,
            dropout=config.dropout_rate,
            batch_first=True
        )
        base_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_bs_decoder_layers
        )
        
        self.transformer_encoder = RoPEWrapper(base_encoder, self.rope)
        
        self.register_buffer(
            'causal_mask_base',
            torch.triu(torch.ones(1000, 1000), diagonal=1).bool(),
            persistent=False
        )
        
        self.out_head = nn.Sequential(
            nn.LayerNorm(config.bs_decoder_dimension),
            nn.Linear(config.bs_decoder_dimension, config.bs_decoder_dimension // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.bs_decoder_dimension // 2, config.out_dim)
        )

        self.blendshape_loss = BlendShapeLoss()

    def get_causal_mask(self, seq_len: int, device):
        if seq_len <= self.causal_mask_base.size(0):
            return self.causal_mask_base[:seq_len, :seq_len].to(device)
        else:
            return torch.triu(
                torch.ones(seq_len, seq_len, device=device), 
                diagonal=1
            ).bool()

    def process_speech_tokens(self, speech_tokens: torch.Tensor) -> tuple:
        device = speech_tokens.device
        speech_mask = speech_tokens.ne(IGNORE_INDEX)
        speech_tokens_clean = speech_tokens.masked_fill(~speech_mask, 0)
        
        speech_emb = self.speech_token_embedding(speech_tokens_clean)
        speech_token_features = self.speech_token_projection(speech_emb)
        speech_token_features = speech_token_features * speech_mask.unsqueeze(-1).to(speech_token_features.dtype)
        
        return speech_token_features, speech_mask

    def forward(
        self,
        speech_gen_hidden_states: torch.FloatTensor,
        speech_tokens: torch.Tensor,
        bs_labels: Optional[torch.FloatTensor] = None,
        bs_lengths: Optional[torch.LongTensor] = None,
        speech_gen_mask: Optional[torch.Tensor] = None
    ):
        # 设备和dtype转换
        target_device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        
        if speech_gen_hidden_states.device != target_device or speech_gen_hidden_states.dtype != target_dtype:
            speech_gen_hidden_states = speech_gen_hidden_states.to(device=target_device, dtype=target_dtype)
        
        if speech_tokens.device != target_device:
            speech_tokens = speech_tokens.to(device=target_device)
        
        if bs_labels is not None and bs_labels.device != target_device:
            bs_labels = bs_labels.to(device=target_device, dtype=target_dtype)
        
        if bs_lengths is not None and bs_lengths.device != target_device:
            bs_lengths = bs_lengths.to(device=target_device)
        
        if speech_gen_mask is not None and speech_gen_mask.device != target_device:
            speech_gen_mask = speech_gen_mask.to(device=target_device)

        device = speech_gen_hidden_states.device
        B = speech_gen_hidden_states.size(0)

        # 确定目标帧数
        if bs_labels is not None:
            target_frames = bs_labels.shape[1]
        else:
            target_frames = units_to_render_frames(
                speech_tokens,
                ctc_hz=12.5, 
                fps=30,
                ignore_index=IGNORE_INDEX,
                rounding='floor', 
                debug=False
            )

        # 特征提取
        speech_gen_features = self.speech_gen_projection(speech_gen_hidden_states)
        speech_token_features, _ = self.process_speech_tokens(speech_tokens)
        
        # TQGF 融合
        if speech_token_features.shape[1] != target_frames:
            speech_token_features_interp = F.interpolate(
                speech_token_features.transpose(1, 2),
                size=target_frames, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        else:
            speech_token_features_interp = speech_token_features
        
        fused_features = self.tqgf(
            text_reps=speech_gen_features,
            query_reps=speech_token_features_interp,
            text_mask=speech_gen_mask
        )
        
        tgt_sequence = fused_features

        T = target_frames
        tgt_mask = self.get_causal_mask(T, device)
        if tgt_mask.dtype.is_floating_point:
            tgt_mask = tgt_mask.to(tgt_sequence.dtype)
        
        if bs_lengths is not None:
            tgt_len = bs_lengths.to(device)
        else:
            raw_len = speech_tokens.ne(IGNORE_INDEX).sum(dim=-1).to(device)
            denom = speech_token_features.shape[1] if speech_token_features.shape[1] > 0 else 1
            approx = (raw_len.float() / float(denom) * T).long()
            tgt_len = torch.clamp(approx, min=1, max=T)
        
        tgt_key_padding_mask = (
            torch.arange(T, device=device)[None, :].expand(B, -1) >= tgt_len[:, None]
        )

        decoded_features = self.transformer_encoder(
            src=tgt_sequence,
            mask=tgt_mask,
            src_key_padding_mask=tgt_key_padding_mask
        )

        blendshape_params = self.out_head(decoded_features)

        if bs_labels is not None:
            loss = self.blendshape_loss(blendshape_params, bs_labels, bs_lengths)
            return loss
        else:
            return blendshape_params

    def predict(self, speech_gen_hidden_states, speech_tokens, speech_gen_mask=None):
        target_device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        
        if speech_gen_hidden_states.device != target_device or speech_gen_hidden_states.dtype != target_dtype:
            speech_gen_hidden_states = speech_gen_hidden_states.to(device=target_device, dtype=target_dtype)
        
        if speech_tokens.device != target_device:
            speech_tokens = speech_tokens.to(device=target_device)
        
        if speech_gen_mask is not None and speech_gen_mask.device != target_device:
            speech_gen_mask = speech_gen_mask.to(device=target_device)

        return self.forward(
            speech_gen_hidden_states=speech_gen_hidden_states,
            speech_tokens=speech_tokens,
            bs_labels=None,
            speech_gen_mask=speech_gen_mask
        )
