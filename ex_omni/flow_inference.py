import torch
import torchaudio
import numpy as np
import re
from hyperpyyaml import load_hyperpyyaml
import uuid
from collections import defaultdict


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


class AudioDecoder:
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path, device="cuda"):
        self.device = device

        with open(config_path, 'r') as f:
            self.scratch_configs = load_hyperpyyaml(f)

        # Load models
        self.flow = self.scratch_configs['flow']
        self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device))
        self.hift = self.scratch_configs['hift']
        self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device))

        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 5
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 1
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

    def token2wav(self, token, uuid, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192), finalize=False):
        tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      embedding=embedding.to(self.device))

        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]

        else:
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            del self.hift_cache_dict[uuid]
            del self.mel_overlap_dict[uuid]
        return tts_speech, tts_mel

    def offline_inference(self, token):
        this_uuid = str(uuid.uuid1())
        tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech.cpu()

    def stream_inference(self, token):
        token.to(self.device)
        this_uuid = str(uuid.uuid1())

        # Prepare other necessary input tensors
        llm_embedding = torch.zeros(1, 192).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32).to(self.device)

        tts_speechs = []
        tts_mels = []

        block_size = self.flow.encoder.block_size
        prev_mel = None

        for idx in range(0, token.size(1), block_size):
            # if idx>block_size: break
            tts_token = token[:, idx:idx + block_size]

            print(tts_token.size())

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                flow_prompt_speech_token = token[:, :idx]

            if idx + block_size >= token.size(-1):
                is_finalize = True
            else:
                is_finalize = False

            tts_speech, tts_mel = self.token2wav(tts_token, uuid=this_uuid,
                                                 prompt_token=flow_prompt_speech_token.to(self.device),
                                                 prompt_feat=prompt_speech_feat.to(self.device), finalize=is_finalize)

            prev_mel = tts_mel
            prev_speech = tts_speech
            print(tts_mel.size())

            tts_speechs.append(tts_speech)
            tts_mels.append(tts_mel)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        return tts_speech.cpu()


    def token2mel(
        self,
        token,              
        uuid=None,
        prompt_token=None,  
        prompt_feat=None,    
        embedding=None,     
        prompt_token_len=None, 
        prompt_feat_len=None,  
        finalize=False,
    ):
        """
        返回:
        feat: [B, L_max, output_size]  生成的全段特征(含prompt段)
        feat_len: [B]  每条样本生成长度(含prompt段)
        prompt_feat_len: [B]  每条样本的prompt长度(若无prompt则为0)
        """
        device = self.device

        B = token.size(0)
        if prompt_token is None:
            prompt_token = torch.zeros(B, 0, dtype=token.dtype, device=token.device)
        if prompt_feat is None:
            prompt_feat = torch.zeros(B, 0, 80, dtype=torch.float32, device=token.device)
        if embedding is None:
            embedding = torch.zeros(B, 192, dtype=torch.float32, device=token.device)
        if prompt_token_len is None:
            prompt_token_len = torch.zeros(B, dtype=torch.int32, device=token.device)
        if prompt_feat_len is None:
            prompt_feat_len = torch.zeros(B, dtype=torch.int32, device=token.device)

        # 统一 dtype/device
        token = token.to(device)
        prompt_token = prompt_token.to(device)
        prompt_feat = prompt_feat.to(device)
        embedding = embedding.to(device)
        token_len = torch.tensor([token.shape[1]] * B, dtype=torch.int32, device=device) if not torch.is_tensor(token) \
            else torch.as_tensor(token_len if 'token_len' in locals() else [token.shape[1]] * B, dtype=torch.int32, device=device)

        # 实际推理
        return self.flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
        )


    def inference(
        self,
        token,                # [B, T_tok]
        token_len,            # [B]
        prompt_token,         # [B, T_ptok]
        prompt_token_len,     # [B]
        prompt_feat,          # [B, L_pfeat, 80]
        prompt_feat_len,      # [B]
        embedding,            # [B, 192]
    ):
        device = token.device
        B = token.size(0)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)   
        # concat text and prompt_text (batch-wise)
        token = torch.concat([prompt_token, token], dim=1)   
        token_len = prompt_token_len + token_len           

        # mask: True=valid
        pad_mask = ~make_pad_mask(token_len, T=token.size(1), device=device)  
        mask = pad_mask.float().unsqueeze(-1).to(embedding)                

        # embedding lookup
        token_int = torch.clamp(token, min=0)             
        token_emb = self.input_embedding(token_int)        
        token_emb = token_emb * mask                       
        # text encode
        h, h_lengths = self.encoder(token_emb, token_len)   
        h = self.encoder_proj(h)                             

        feat_len = ((token_len.to(torch.float32) / self.input_frame_rate) * (22050.0 / 256.0)).to(torch.int32)  # [B]

        h, h_lengths = self.length_regulator(h, feat_len)   

        L_max = int(feat_len.max().item())
        conds = torch.zeros([B, L_max, self.output_size], device=device)  
        if prompt_feat.size(1) != 0:
            for i in range(B):
                li = int(prompt_feat_len[i].item())
                if li > 0:
                    conds[i, :li, :] = prompt_feat[i, :li, :]

        conds = conds.transpose(1, 2)                      

        mask = ~make_pad_mask(feat_len, T=L_max, device=device) 
        mask = mask.unsqueeze(1)                                 

        # 解码
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),  
            mask=mask,                         
            spks=embedding,                    
            cond=conds,                      
            n_timesteps=10
        )                                        
        feat = feat.transpose(1, 2).contiguous() 

        return feat, feat_len, prompt_feat_len


def make_pad_mask(lengths: torch.Tensor, T: int = None, device=None):
    """
    lengths: [B] int32/int64
    T: optional, if None will use lengths.max()
    return: [B, T] bool, True = padded 位置
    """
    device = device if device is not None else lengths.device
    B = lengths.size(0)
    if T is None:
        T = int(lengths.max().item())
    arange = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
    # padded = positions >= length
    pad_mask = arange >= lengths.unsqueeze(1)  # True=pad
    return pad_mask
