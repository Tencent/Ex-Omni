import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen3ForCausalLM, Qwen3Config
from safetensors.torch import load_file
from ex_omni.constants import IGNORE_INDEX
import time

# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Label smoothing module."""

import torch
from torch import nn

import torch
from einops import rearrange, repeat
from torch import einsum, nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

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
    def __init__(self, dim=896, dim_head=256, heads=16):
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
        
        # Gated Attention: head-specific elementwise (论文 Table 1 row 5)
        self.gate_proj = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, text_reps, query_reps, text_mask=None):
        text_reps_normed = self.norm_text(text_reps)
        query_reps_normed = self.norm_query(query_reps)
        
        h, d = self.heads, self.dim_head
        
        # QKV
        q = rearrange(self.to_q(query_reps_normed), "b n (h d) -> b h n d", h=h, d=d)
        k, v = self.to_kv(text_reps_normed).chunk(2, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=h, d=d)
        v = rearrange(v, "b n (h d) -> b h n d", h=h, d=d)
        
        # Attention
        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        
        if text_mask is not None:
            mask = rearrange(text_mask, "b n -> b 1 1 n")
            sim = sim.masked_fill(~mask, float('-inf'))
        
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)  # [B, H, N, Dh]
        
        # Gated Attention (SDPA output, G1, head-specific elementwise)
        gate_scores = self.gate_proj(query_reps_normed)  # [B, N, H*Dh]
        gate_scores = rearrange(gate_scores, "b n (h d) -> b h n d", h=h, d=d)  # [B, H, N, Dh]
        gate_scores = torch.sigmoid(gate_scores)
        out = out * gate_scores  # Head-specific elementwise gating
        
        # Output projection
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
    

class TQGF(nn.Module):
    """Text-Query Guided Fusion with gated cross-attention."""

    def __init__(
        self,
        dim,
        depth=2,
        dim_head=256,
        heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TQGFGatedCrossAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_reps, query_reps, text_mask=None):
        b,n,_=text_reps.shape
        for attn, ff in self.layers:
            query_reps = attn(text_reps, query_reps, text_mask=text_mask) + query_reps
            query_reps = ff(query_reps) + query_reps
        return self.norm(query_reps)

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute loss between x and target.
        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor): target signal masked with self.padding_id (batch, seqlen)
            reduction (str): 'mean', 'sum', or 'none'
        Returns:
            loss (torch.Tensor): The KL loss
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        x = x.reshape(-1, self.size)
        target = target.reshape(-1)
        
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        
        # 重新reshape为 (batch_size, seq_len)
        kl = kl.sum(dim=1)  # sum over vocab dimension if needed
        kl = kl.view(batch_size, seq_len)
        ignore = ignore.view(batch_size, seq_len)
        
        # 对每个样本计算loss
        kl_masked = kl.masked_fill(ignore, 0)
        
        if reduction == 'none':
            # 返回每个样本的loss
            if self.normalize_length:
                # 每个样本除以其有效长度
                valid_lengths = (~ignore).sum(dim=1).float()
                valid_lengths = torch.clamp(valid_lengths, min=1)  # 避免除零
                return kl_masked.sum(dim=1) / valid_lengths
            else:
                return kl_masked.sum(dim=1)
        elif reduction == 'sum':
            return kl_masked.sum()
        else:  # reduction == 'mean'
            denom = total if self.normalize_length else batch_size
            return kl_masked.sum() / denom

# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

def _uniform_assignment(src_lens, tgt_lens):
    tgt_indices = torch.arange(torch.max(tgt_lens)).expand(len(tgt_lens), -1).to(tgt_lens.device)
    ratio = tgt_lens / src_lens
    index_t = (tgt_indices / ratio.view(-1, 1)).long()
    return index_t

def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()

class Qwen3Encoder(torch.nn.Module):
    def __init__(self, pretrain_path, attn_implementation="sdpa"):
        super().__init__()
        qwen_config = Qwen3Config.from_pretrained(pretrain_path)
        qwen_config.attn_implementation = attn_implementation
        qwen_config._attn_implementation = attn_implementation
        self.model = Qwen3ForCausalLM(qwen_config)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

class SpeechGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers, n_dims, n_heads, n_inter_dims = list(map(int, config.ctc_decoder_config[1:-1].split(",")))
        _config = copy.deepcopy(config)
        _config.hidden_size = n_dims
        _config.num_hidden_layers = n_layers
        _config.num_attention_heads = n_heads
        _config.num_key_value_heads = n_heads
        _config.intermediate_size = n_inter_dims
        _config._attn_implementation = getattr(config, 'attn_implementation', 'sdpa')

        config.speech_gen_hidden_size = Qwen3Config.from_pretrained(config.pretrain_speech_generator_weights).hidden_size
        self.upsample_factor = config.ctc_upsample_factor
        self.unit_vocab_size = config.unit_vocab_size
        self.llm = Qwen3Encoder(
            config.pretrain_speech_generator_weights,
            attn_implementation=getattr(config, "attn_implementation", "sdpa"),
        )
        
        self.n_dims=1024 # fix

        self.llm_input_size = self.n_dims
        self.llm_output_size = self.n_dims

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, self.llm_input_size)
        self.llm_decoder = nn.Linear(self.llm_output_size, self.unit_vocab_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.unit_vocab_size + 3,
            padding_idx=IGNORE_INDEX,
            smoothing=0,
            normalize_length=True,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(self.unit_vocab_size + 3, self.llm_input_size)

        # 4. sampling method
        self.sampling = ras_sampling

        modules = [
        nn.Linear(config.hidden_size, self.n_dims * 4),
        # nn.LayerNorm(self.n_dims * 4),  
        nn.GELU(),
        nn.Linear(self.n_dims * 4, self.n_dims * 4),
        # nn.LayerNorm(self.n_dims * 4),   
        ]
        
        self.input_proj = nn.Sequential(*modules)

        self.tqgf = TQGF(self.n_dims)


    def sampling_ids(
            self,
            weighted_scores,
            decoded_tokens,
            sampling,
            ignore_eos
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.unit_vocab_size not in top_ids):
                break
        return top_ids

    def forward(self, text_reps, text_labels, speech_tokens, text_tokens=None, embedding=None):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        text_reps = text_reps.to(device=device, dtype=dtype)
        text_labels = text_labels.to(device=device)
        speech_tokens = speech_tokens.to(device=device)
        if text_tokens is not None:
            text_tokens = text_tokens.to(device=device)
        
        # 处理text representations
        tgt_text_reps = [text_rep[text_label.ne(IGNORE_INDEX)] for text_rep, text_label in zip(text_reps, text_labels)]
        text_reps_lens = torch.LongTensor([len(rep)*4 for rep in tgt_text_reps]).to(device)

        tgt_text_reps_padding_mask = ~lengths_to_padding_mask(text_reps_lens)
        
        tgt_text_reps = torch.nn.utils.rnn.pad_sequence(tgt_text_reps, batch_first=True)
        tgt_text_reps = self.input_proj(tgt_text_reps)
        tgt_text_reps = rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)
        
        # 计算token长度
        text_token_lens = text_tokens.ne(IGNORE_INDEX).long().sum(dim=-1)
        speech_token_lens = speech_tokens.ne(IGNORE_INDEX).long().sum(dim=-1)
        
        # 预处理有效tokens
        valid_text_tokens = [x[x.ne(IGNORE_INDEX)] for x in text_tokens]
        valid_speech_tokens = [x[x.ne(IGNORE_INDEX)] for x in speech_tokens]
        
        # 获取embeddings
        text = torch.nn.utils.rnn.pad_sequence(valid_text_tokens, batch_first=True, padding_value=0)
        text_embeddings = self.llm.model.model.embed_tokens(text)
        text_embeddings = self.tqgf(tgt_text_reps, text_embeddings, tgt_text_reps_padding_mask)
        
        speech = torch.nn.utils.rnn.pad_sequence(valid_speech_tokens, batch_first=True, padding_value=0)
        speech_embeddings = self.speech_embedding(speech)
        
        # 预计算特殊token embeddings
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(-1)  # 直接flatten
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(-1)
        
        # 构建输入和标签
        lm_inputs = []
        lm_targets = []
        
        for i, (text_len, speech_len) in enumerate(zip(text_token_lens, speech_token_lens)):
            # 构建输入embedding序列
            input_seq = torch.cat([
                sos_eos_emb.unsqueeze(0),
                text_embeddings[i][:text_len],
                task_id_emb.unsqueeze(0),
                speech_embeddings[i][:speech_len],
                sos_eos_emb.unsqueeze(0)
            ])
            lm_inputs.append(input_seq)
            
            # 构建标签序列
            target_seq = torch.tensor(
                [IGNORE_INDEX] * (1 + text_len + 1) + 
                valid_speech_tokens[i][:speech_len].tolist() + 
                [self.unit_vocab_size],
                dtype=torch.long,
                device=speech_tokens.device
            )
            lm_targets.append(target_seq)
        
        # 批量填充
        max_len = max(x.shape[0] for x in lm_inputs)
        batch_size = len(lm_inputs)
        embed_dim = lm_inputs[0].shape[1]
        
        # 一次性创建所有张量
        input_embeds = torch.zeros((batch_size, max_len, embed_dim), 
                                dtype=lm_inputs[0].dtype, device=lm_inputs[0].device)
        labels = torch.full((batch_size, max_len), IGNORE_INDEX, 
                        dtype=torch.long, device=speech_tokens.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=speech_tokens.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=speech_tokens.device)
        
        # 填充数据
        for i, (input_seq, target_seq) in enumerate(zip(lm_inputs, lm_targets)):
            seq_len = input_seq.shape[0]
            input_embeds[i, :seq_len] = input_seq
            labels[i, :seq_len] = target_seq
            attention_mask[i, :seq_len] = True
            position_ids[i, :seq_len] = torch.arange(seq_len, device=speech_tokens.device)
        
        # 模型前向传播
        outputs = self.llm.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        
        logits = self.llm_decoder(outputs.hidden_states[-1])
        loss = self.criterion_ce(logits[:, :-1], labels[:, 1:], 'none')
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    # def predict(self, 
    #         tgt_text_reps, 
    #         text,
    #         sampling= 25,
    #         max_token_text_ratio = 20,
    #         min_token_text_ratio = 2,
    #         **kwargs):
 
    #     target_device = next(self.input_proj.parameters()).device
    #     target_dtype = next(self.input_proj.parameters()).dtype
    #     # 打印调试信息
    #     print(f"[DEBUG] predict 方法被调用")
    #     print(f"  - tgt_text_reps device: {tgt_text_reps.device}, dtype: {tgt_text_reps.dtype}")
    #     print(f"  - text device: {text.device}")
    #     print(f"  - target_device (input_proj): {target_device}")
    #     print(f"  - target_dtype: {target_dtype}")

    #     # 将输入移到正确的设备
    #     if tgt_text_reps.device != target_device or tgt_text_reps.dtype != target_dtype:
    #         print(f"  - 转换 tgt_text_reps: {tgt_text_reps.device} -> {target_device}")
    #         tgt_text_reps = tgt_text_reps.to(device=target_device, dtype=target_dtype)
    #     if text.device != target_device:
    #         print(f"  - 转换 text: {text.device} -> {target_device}")
    #         text = text.to(device=target_device)

        
    #     # 1. 处理文本表示
    #     tgt_text_reps = self.input_proj(tgt_text_reps)
    #     tgt_text_reps=rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)
    #     text=text[:,:-1]
    #     text_len=text.size(1)      
    #     text_embedding = self.llm.model.model.embed_tokens(text)
    #     text_embedding = self.tqgf(tgt_text_reps,text_embedding, None)

    #     # 2. encode embedding
    #     embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(text.device)

    #     # 3. concat llm_input
    #     sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
    #     task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

    #     lm_input = torch.concat([sos_eos_emb, embedding, text_embedding, task_id_emb], dim=1)

    #     # 4. cal min/max_length
    #     min_len = int(text_len * min_token_text_ratio)
    #     max_len = int(text_len * max_token_text_ratio)

    #     # print(min_len,max_len)

    #     # 5. step by step decode
    #     out_tokens = []
    #     cache = None
    #     for i in range(max_len):
    #         y_pred, cache = self.llm.forward_one_step(lm_input,
    #                                                 masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
    #                                                 cache=cache)
    #         logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
    #         # import pdb
    #         # pdb.set_trace()
    #         top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
    #         if top_ids == self.unit_vocab_size:
    #             break
    #         if top_ids > self.unit_vocab_size:
    #             continue
    #         # in stream mode, yield token one by one
    #         # yield top_ids
    #         out_tokens.append(top_ids)
    #         lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            
    #     return ' '.join([str(x) for x in out_tokens]), out_tokens

    def predict(self, 
            tgt_text_reps, 
            text,
            sampling=25,
            max_token_text_ratio=20,
            min_token_text_ratio=2,
            return_hidden_states=True,  # 新增参数
            **kwargs):

        # 1. 处理文本表示
        tgt_text_reps = self.input_proj(tgt_text_reps)
        tgt_text_reps = rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)
        
        # drop the eos token
        text = text[:, :-1]
        text_len = text.size(1)      
        text_embedding = self.llm.model.model.embed_tokens(text)
        text_embedding = self.tqgf(tgt_text_reps, text_embedding, None)

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(text.device)

        # 3. concat llm_input (这就是初始prompt)
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        lm_input = torch.concat([sos_eos_emb, embedding, text_embedding, task_id_emb], dim=1)

        # 4. cal min/max_length
        min_len = int(text_len * min_token_text_ratio)
        max_len = int(text_len * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        all_hidden_states = []  # 收集所有步骤的hidden states
        cache = None
        
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                cache=cache
            )
            
            # 保存hidden states
            if return_hidden_states:
                if i == 0:
                    # 第0步（prompt阶段）：保存最后一个位置
                    # 对应 hidden_states[0][-1][:, -1:, :]
                    all_hidden_states.append(y_pred[:, -1:, :])  # [1, 1, hidden_dim]
                else:
                    # 后续生成步骤：保存整个输出（单个token）
                    # 对应 hidden_states[i][-1]
                    all_hidden_states.append(y_pred)  # [1, 1, hidden_dim]
            
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0), 
                out_tokens, 
                sampling, 
                ignore_eos=True if i < min_len else False
            ).item()
            
            if top_ids == self.unit_vocab_size:
                break
            if top_ids > self.unit_vocab_size:
                continue
                
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        if return_hidden_states and len(all_hidden_states) > 0:
            hidden_states = torch.cat(all_hidden_states, dim=1)
            
            if hidden_states.device != tgt_text_reps.device:
                hidden_states = hidden_states.to(device=tgt_text_reps.device,dtype=tgt_text_reps.dtype)
            
            return out_tokens, hidden_states
        else:
            return ' '.join([str(x) for x in out_tokens]), out_tokens

    # def predict( 
    #         self, 
    #         tgt_text_reps, 
    #         text,
    #         sampling=25,
    #         max_token_text_ratio=20,
    #         min_token_text_ratio=2,
    #         return_hidden_states=True,  # 新增参数
    #         **kwargs):
    #     """
    #     AR 语音 token 解码 + 可选 hidden_states 返回，并在内部记录：
    #         - self.last_speech_gen_time : 整个解码总耗时
    #         - self.last_ttft            : Time-To-First-Token
    #         - self.last_speech_audio_dur: 当前先置为 None（如果之后有 unit->秒 的映射可以再填）
    #     """

    #     # ===== 时间统计初始化 =====
    #     decode_start = time.time()
    #     first_token_time = None

    #     # 1. 处理文本表示
    #     tgt_text_reps = self.input_proj(tgt_text_reps)
    #     tgt_text_reps = rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)
        
    #     # drop the eos token
    #     text = text[:, :-1]
    #     text_len = text.size(1)      
    #     text_embedding = self.llm.model.model.embed_tokens(text)
    #     text_embedding = self.tqgf(tgt_text_reps, text_embedding, None)

    #     # 2. encode embedding
    #     embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(text.device)

    #     # 3. concat llm_input (这就是初始prompt)
    #     sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
    #     task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

    #     lm_input = torch.concat([sos_eos_emb, embedding, text_embedding, task_id_emb], dim=1)

    #     # 4. cal min/max_length
    #     min_len = int(text_len * min_token_text_ratio)
    #     max_len = int(text_len * max_token_text_ratio)

    #     # 5. step by step decode
    #     out_tokens = []
    #     all_hidden_states = []  # 收集所有步骤的hidden states
    #     cache = None
        
    #     for i in range(max_len):
    #         y_pred, cache = self.llm.forward_one_step(
    #             lm_input,
    #             masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
    #             cache=cache
    #         )
            
    #         # 保存hidden states
    #         if return_hidden_states:
    #             if i == 0:
    #                 # 第0步（prompt阶段）：保存最后一个位置
    #                 all_hidden_states.append(y_pred[:, -1:, :])  # [1, 1, hidden_dim]
    #             else:
    #                 # 后续生成步骤：保存整个输出（单个token）
    #                 all_hidden_states.append(y_pred)  # [1, 1, hidden_dim]
            
    #         logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
    #         top_ids = self.sampling_ids(
    #             logp.squeeze(dim=0), 
    #             out_tokens, 
    #             sampling, 
    #             ignore_eos=True if i < min_len else False
    #         ).item()
            
    #         if top_ids == self.unit_vocab_size:
    #             break
    #         if top_ids > self.unit_vocab_size:
    #             continue

    #         # 第一次成功生成 token，记录 TTFT
    #         if first_token_time is None:
    #             first_token_time = time.time()
                
    #         out_tokens.append(top_ids)
    #         lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    #     # ===== 结束计时，写入缓存属性 =====
    #     decode_end = time.time()
    #     self.last_speech_gen_time = float(decode_end - decode_start)
    #     self.last_ttft = (
    #         float(first_token_time - decode_start) if first_token_time is not None else None
    #     )

    #     # 这里暂时没有“生成语音的真实时长”，先留空，
    #     # 如果你之后在 TTS 解码那边能拿到秒数，可以在外层补回到 profile 里。
    #     self.last_speech_audio_dur = None

    #     # ===== 返回值保持原样：out_tokens & hidden_states =====
    #     if return_hidden_states and len(all_hidden_states) > 0:
    #         hidden_states = torch.cat(all_hidden_states, dim=1)
            
    #         if hidden_states.device != tgt_text_reps.device:
    #             hidden_states = hidden_states.to(device=tgt_text_reps.device, dtype=tgt_text_reps.dtype)
            
    #         return out_tokens, hidden_states
    #     else:
    #         return ' '.join([str(x) for x in out_tokens]), out_tokens
