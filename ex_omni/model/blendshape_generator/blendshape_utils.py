# Borrowed from https://github.com/EvelynFan/FaceFormer/blob/main/main.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ex_omni.constants import IGNORE_INDEX
import numpy as np
import math
import torch

def units_to_render_frames(
    speech_units: torch.Tensor,
    *,
    ctc_hz: float | None = 12.5,   
    sr: int | None = None,      
    hop: int | None = None,
    fps: int = 30,
    ignore_index: int = -100,     
    rounding: str = "floor",     
    allow_zero: bool = True,
    debug: bool = True,
) -> int:
    x = speech_units.reshape(-1)
    valid = (x > 0) & x.ne(ignore_index)
    n_units = int(valid.sum().item())

    if debug:
        cnt_ne_ignore = int(x.ne(ignore_index).sum().item())
        cnt_ne0       = int(x.ne(0).sum().item())
        cnt_pos       = int((x > 0).sum().item())

    if n_units <= 0:
        return 0 if allow_zero else 1

    if ctc_hz is not None:
        unit_sec = 1.0 / float(ctc_hz)
    elif (sr is not None) and (hop is not None):
        unit_sec = hop / float(sr)
    else:
        raise ValueError("需要提供 ctc_hz 或者 (sr, hop) 之一。")

    total_sec = n_units * unit_sec

    raw_frames = total_sec * fps
    if rounding == "floor":
        n_frames = int(math.floor(raw_frames))
    elif rounding == "ceil":
        n_frames = int(math.ceil(raw_frames))
    else:
        n_frames = int(round(raw_frames))

    if not allow_zero:
        n_frames = max(n_frames, 1)

    if debug:
        print(f"[DBG] ctc_hz={ctc_hz}, unit_sec={unit_sec:.6f}, total_sec={total_sec:.6f}, "
              f"fps={fps}, raw_frames={raw_frames:.3f}, n_frames={n_frames}")

        est_hz = (n_units * fps) / max(n_frames, 1)
        print(f"[DBG] est_hz_from_result≈{est_hz:.4f} (应接近设置的 ctc_hz)")

    return n_frames


def pad_to_target_len(x, target):
    B, T_pred, D = x.shape
    T_gt = target.shape[1]
    if T_pred < T_gt:
        pad_len = T_gt - T_pred
        pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)   # [B, T_gt, D]
    elif T_pred > T_gt:
        x = x[:, :T_gt, :]
    return x


class BlendShapeLoss(nn.Module):
    def __init__(self, velocity_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.velocity_weight = velocity_weight
        self.eps = eps

    def forward(self, x: torch.Tensor, target: torch.Tensor, target_length: torch.Tensor):
        B, T_gt, D = target.shape
        x = pad_to_target_len(x, target)     # 先 pad/cut 到 target 的长度
        T = T_gt

        x = x.float(); target = target.float()
        target_length = target_length.to(x.device).float().clamp(max=T)

        # mask
        t_idx = torch.arange(T, device=x.device)[None, :]          # [1, T]
        mask = (t_idx < target_length[:, None]).float()            # [B, T]

        # 1) MSE
        mse_per_step = self.mse(x, target).mean(-1) * mask
        L_mse_per_seq = mse_per_step.sum(1) / (mask.sum(1) + self.eps)
        L_mse = L_mse_per_seq.mean()

        # 2) Velocity (when the amount of data is sufficient, it doesn't have much effect.)
        # if T > 1:
        #     pred_vel = x[:, 1:] - x[:, :-1]
        #     target_vel = target[:, 1:] - target[:, :-1]
        #     mask_vel = mask[:, 1:] * mask[:, :-1]
        #     vel_step = self.mse(pred_vel, target_vel).mean(-1) * mask_vel
        #     L_vel_per_seq = vel_step.sum(1) / (mask_vel.sum(1) + self.eps)
        #     L_vel = L_vel_per_seq.mean()
        # else:
        #     L_vel = x.new_tensor(0.0)
        L_vel = x.new_tensor(0.0)

        total = L_mse + self.velocity_weight * L_vel
        return total