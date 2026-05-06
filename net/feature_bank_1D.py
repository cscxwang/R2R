import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
class LayerNorm3D(nn.Module):
    """
    LayerNorm for 5D features.
    Input shape: [1, C, N, H, W], where the leading 1 is not the batch size.
    The normalization dimension can be 'C' or 'N'.
    """
    def __init__(self, normalized_dim, num_features, eps: float = 1e-6):
        """
        Args:
            normalized_dim: 'C' or 'N'
            num_features: size of the normalized dimension
            eps: numerical stability term
        """
        super().__init__()
        assert normalized_dim in ['C', 'B'], "normalized_dim must be 'C' or 'B'"
        self.normalized_dim = normalized_dim
        self.eps = eps
        self.layer_norm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, x):
        # x: [1, C, N, H, W]
        B, C, N, H, W = x.shape
        x = x.squeeze(0)
        if self.normalized_dim == 'C':
            x_perm = x.permute(1,2,3,0)  # [N,H,W,C]
            x_norm = self.layer_norm(x_perm)
            x_out = x_norm.permute(3,0,1,2)  # [1,C,N,H,W]
        else:
            x_perm = x.permute(0,2,3,1)  # [C,H,W,N]
            x_norm = self.layer_norm(x_perm)
            x_out = x_norm.permute(0,3,1,2)  # [1,C,N,H,W]

        return x_out.unsqueeze(0)
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class DegradationMemory(nn.Module):
    def __init__(self,
                 opt,
                 T_max=32,
                 key_dim=64,
                 value_dim=512,
                 denoise_bank_key=None,
                 denoise_bank_value=None,
                 derain_bank_key=None,
                 derain_bank_value=None,
                 dehaze_bank_key=None,
                 dehaze_bank_value=None,
                 deblur_bank_key=None,
                 deblur_bank_value=None,
                 lowlight_bank_key=None,
                 lowlight_bank_value=None,
                 deg_types=["denoise_15"]                 ):
        super().__init__()
        self.deg_types = deg_types
        self.T_max = T_max
        self.update_step = T_max//5
        self.key_dim=key_dim
        self.value_dim=value_dim

        if "denoise" in self.deg_types[0]:
            self.update_denoise_key_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(self.update_step, 3, 3), padding=(0, 1, 1), groups=key_dim),
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(key_dim),
                nn.ReLU(),

                )
            self.update_denoise_value_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(self.update_step, 3, 3), padding=(0, 1, 1), groups=value_dim),
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(value_dim),
                nn.ReLU(),
            )
            print("single task: denoise")
        if "derain" in self.deg_types[0]:
            self.update_derain_key_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(self.update_step, 3, 3),
                          padding=(0, 1, 1), groups=key_dim),
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(key_dim),
                nn.ReLU(),
            )
            self.update_derain_value_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(self.update_step, 3, 3),
                          padding=(0, 1, 1), groups=value_dim),
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(value_dim),
                nn.ReLU(),
            )
            print("single task: derain")
        if "dehaze" in self.deg_types[0]:
            self.update_dehaze_key_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(self.update_step, 3, 3),
                          padding=(0, 1, 1), groups=key_dim),
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(key_dim),
                nn.ReLU(),
            )
            self.update_dehaze_value_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(self.update_step, 3, 3),
                          padding=(0, 1, 1), groups=value_dim),
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(value_dim),
                nn.ReLU(),
            )
            print("single task: dehaze")
        if "deblur" in self.deg_types[0]:
            self.update_deblur_key_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(self.update_step, 3, 3),
                          padding=(0, 1, 1), groups=key_dim),
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(key_dim),
                nn.ReLU(),
            )
            self.update_deblur_value_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(self.update_step, 3, 3),
                          padding=(0, 1, 1), groups=value_dim),
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(value_dim),
                nn.ReLU(),
            )
            print("single task: deblur")
        if "lowlight" in self.deg_types[0]:
            self.update_lowlight_key_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(self.update_step, 3, 3),
                          padding=(0, 1, 1), groups=key_dim),
                nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(key_dim),
                nn.ReLU(),
            )
            self.update_lowlight_value_3Dconv = nn.Sequential(
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(self.update_step, 3, 3),
                          padding=(0, 1, 1), groups=value_dim),
                nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                nn.BatchNorm3d(value_dim),
                nn.ReLU(),
            )
            print("single task: lowlight")

        self.init_bank(denoise_bank_key, denoise_bank_value, derain_bank_key, derain_bank_value, dehaze_bank_key, dehaze_bank_value, deblur_bank_key, deblur_bank_value, lowlight_bank_key, lowlight_bank_value)

    def init_bank(self,
                  denoise_bank_key=None,
                  denoise_bank_value=None,
                  derain_bank_key=None,
                  derain_bank_value=None,
                  dehaze_bank_key=None,
                  dehaze_bank_value=None,
                  deblur_bank_key=None,
                  deblur_bank_value=None,
                  lowlight_bank_key=None,
                  lowlight_bank_value=None):
        self.denoise_bank_key = denoise_bank_key if denoise_bank_key is not None else []
        self.denoise_bank_value = denoise_bank_value if denoise_bank_value is not None else []

        self.derain_bank_key = derain_bank_key if derain_bank_key is not None else []
        self.derain_bank_value = derain_bank_value if derain_bank_value is not None else []

        self.dehaze_bank_key = dehaze_bank_key if dehaze_bank_key is not None else []
        self.dehaze_bank_value = dehaze_bank_value if dehaze_bank_value is not None else []

        self.deblur_bank_key = deblur_bank_key if deblur_bank_key is not None else []
        self.deblur_bank_value = deblur_bank_value if deblur_bank_value is not None else []

        self.lowlight_bank_key = lowlight_bank_key if lowlight_bank_key is not None else []
        self.lowlight_bank_value = lowlight_bank_value if lowlight_bank_value is not None else []




    def update_bank(self, deg_type, mk, mv):
        if deg_type == "denoise":

            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            k_list = [mk for _ in range(n)]
            mk = torch.cat(k_list, dim=0)  # [B*n, 512, H, W]

            v_list = [mv for _ in range(n)]
            mv = torch.cat(v_list, dim=0)  # [B*n, 512, H, W]

            for i in range(mk.shape[0]):
                if len(self.denoise_bank_key) < self.T_max:
                    self.denoise_bank_key.append(mk[i:i+1])
                    self.denoise_bank_value.append(mv[i:i+1])
                else:
                    indices = list(range(len(self.denoise_bank_key)))
                    random.shuffle(indices)
                    shuffled_key = [self.denoise_bank_key[j] for j in indices]
                    shuffled_value = [self.denoise_bank_value[j] for j in indices]
                    key_combined = torch.cat(shuffled_key, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 64, T_max, h, w]
                    value_combined = torch.cat(shuffled_value, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 512, T_max,  h, w]
                    self.denoise_bank_key.clear()
                    self.denoise_bank_value.clear()
                    update_keys = self.update_denoise_key_3Dconv(key_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    update_values = self.update_denoise_value_3Dconv(value_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 512, update_step, h, w]
                    self.denoise_bank_key = [update_keys[i:i+1] for i in range(update_keys.shape[0])]
                    self.denoise_bank_value = [update_values[i:i+1] for i in range(update_keys.shape[0])]
                    return
        elif deg_type == "derain":
            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            k_list = [mk for _ in range(n)]
            mk = torch.cat(k_list, dim=0)  # [B*n, 512, H, W]

            v_list = [mv for _ in range(n)]
            mv = torch.cat(v_list, dim=0)  # [B*n, 512, H, W]
            for i in range(mk.shape[0]):
                if len(self.derain_bank_key) < self.T_max:
                    self.derain_bank_key.append(mk[i:i+1])
                    self.derain_bank_value.append(mv[i:i+1])
                else:
                    indices = list(range(len(self.derain_bank_key)))
                    random.shuffle(indices)
                    shuffled_key = [self.derain_bank_key[j] for j in indices]
                    shuffled_value = [self.derain_bank_value[j] for j in indices]
                    key_combined = torch.cat(shuffled_key, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 64, T_max, h, w]
                    value_combined = torch.cat(shuffled_value, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 512, T_max,  h, w]
                    self.derain_bank_key.clear()
                    self.derain_bank_value.clear()
                    update_keys = self.update_derain_key_3Dconv(key_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    update_values = self.update_derain_value_3Dconv(value_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    self.derain_bank_key = [update_keys[i:i+1] for i in range(update_keys.shape[0])]
                    self.derain_bank_value = [update_values[i:i+1] for i in range(update_keys.shape[0])]
                    return
        elif deg_type == "dehaze":
            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            k_list = [mk for _ in range(n)]
            mk = torch.cat(k_list, dim=0)  # [B*n, 512, H, W]

            v_list = [mv for _ in range(n)]
            mv = torch.cat(v_list, dim=0)  # [B*n, 512, H, W]
            for i in range(mk.shape[0]):
                if len(self.dehaze_bank_key) < self.T_max:
                    self.dehaze_bank_key.append(mk[i:i+1])
                    self.dehaze_bank_value.append(mv[i:i+1])

                else:
                    indices = list(range(len(self.dehaze_bank_key)))
                    random.shuffle(indices)
                    shuffled_key = [self.dehaze_bank_key[j] for j in indices]
                    shuffled_value = [self.dehaze_bank_value[j] for j in indices]
                    key_combined = torch.cat(shuffled_key, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 64, T_max, h, w]
                    value_combined = torch.cat(shuffled_value, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 512, T_max,  h, w]
                    self.dehaze_bank_key.clear()
                    self.dehaze_bank_value.clear()
                    update_keys = self.update_dehaze_key_3Dconv(key_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    update_values = self.update_dehaze_value_3Dconv(value_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    self.dehaze_bank_key = [update_keys[i:i+1] for i in range(update_keys.shape[0])]
                    self.dehaze_bank_value = [update_values[i:i+1] for i in range(update_keys.shape[0])]
                    return
        elif deg_type == "deblur":
            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            k_list = [mk for _ in range(n)]
            mk = torch.cat(k_list, dim=0)  # [B*n, 512, H, W]

            v_list = [mv for _ in range(n)]
            mv = torch.cat(v_list, dim=0)  # [B*n, 512, H, W]
            for i in range(mk.shape[0]):
                if len(self.deblur_bank_key) < self.T_max:
                    self.deblur_bank_key.append(mk[i:i+1])
                    self.deblur_bank_value.append(mv[i:i+1])

                else:
                    indices = list(range(len(self.deblur_bank_key)))
                    random.shuffle(indices)
                    shuffled_key = [self.deblur_bank_key[j] for j in indices]
                    shuffled_value = [self.deblur_bank_value[j] for j in indices]
                    key_combined = torch.cat(shuffled_key, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 64, T_max, h, w]
                    value_combined = torch.cat(shuffled_value, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 512, T_max,  h, w]
                    self.deblur_bank_key.clear()
                    self.deblur_bank_value.clear()
                    update_keys = self.update_deblur_key_3Dconv(key_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    update_values = self.update_deblur_value_3Dconv(value_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    self.deblur_bank_key = [update_keys[i:i+1] for i in range(update_keys.shape[0])]
                    self.deblur_bank_value = [update_values[i:i+1] for i in range(update_keys.shape[0])]
                    return
        elif deg_type == "lowlight":
            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            k_list = [mk for _ in range(n)]
            mk = torch.cat(k_list, dim=0)  # [B*n, 512, H, W]

            v_list = [mv for _ in range(n)]
            mv = torch.cat(v_list, dim=0)  # [B*n, 512, H, W]
            for i in range(mk.shape[0]):
                if len(self.lowlight_bank_key) < self.T_max:
                    self.lowlight_bank_key.append(mk[i:i+1])
                    self.lowlight_bank_value.append(mv[i:i+1])

                else:
                    indices = list(range(len(self.lowlight_bank_key)))
                    random.shuffle(indices)
                    shuffled_key = [self.lowlight_bank_key[j] for j in indices]
                    shuffled_value = [self.lowlight_bank_value[j] for j in indices]
                    key_combined = torch.cat(shuffled_key, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 64, T_max, h, w]
                    value_combined = torch.cat(shuffled_value, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 512, T_max,  h, w]
                    self.lowlight_bank_key.clear()
                    self.lowlight_bank_value.clear()
                    update_keys = self.update_lowlight_key_3Dconv(key_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    update_values = self.update_lowlight_value_3Dconv(value_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    self.lowlight_bank_key = [update_keys[i:i+1] for i in range(update_keys.shape[0])]
                    self.lowlight_bank_value = [update_values[i:i+1] for i in range(update_keys.shape[0])]
                    return


    def get_deg_prompt(self, qk, interact_label=None):
        if len(self.denoise_bank_key) != 0 :
            mk = torch.cat(self.denoise_bank_key, dim=0)
            mv = torch.cat(self.denoise_bank_value, dim=0)
            n = len(self.denoise_bank_key)
        if len(self.derain_bank_key) != 0:
            mk = torch.cat(self.derain_bank_key, dim=0)
            mv = torch.cat(self.derain_bank_value, dim=0)
            n = len(self.derain_bank_key)
        if len(self.dehaze_bank_key) != 0 :
            mk = torch.cat(self.dehaze_bank_key, dim=0)
            mv = torch.cat(self.dehaze_bank_value, dim=0)
            n = len(self.dehaze_bank_key)
        if len(self.deblur_bank_key) != 0 :
            mk = torch.cat(self.deblur_bank_key, dim=0)
            mv = torch.cat(self.deblur_bank_value, dim=0)
            n = len(self.deblur_bank_key)
        if len(self.lowlight_bank_key) != 0 :
            mk = torch.cat(self.lowlight_bank_key, dim=0)
            mv = torch.cat(self.lowlight_bank_value, dim=0)
            n = len(self.lowlight_bank_key)

        readout = self.comprehensive_attention_processing(mk, qk, mv, n)


        return readout

    def clear_grad(self, stage=0):
        if stage == 0:
            self.denoise_bank_key.clear()
            self.dehaze_bank_key.clear()
            self.derain_bank_key.clear()
            self.deblur_bank_key.clear()
            self.lowlight_bank_key.clear()

            self.dehaze_bank_value.clear()
            self.denoise_bank_value.clear()
            self.derain_bank_value.clear()
            self.deblur_bank_value.clear()
            self.lowlight_bank_value.clear()

            return
        kv_compress = self.T_max - self.update_step + 1
        if len(self.denoise_bank_key) != 0 :
            self.denoise_bank_key = self.denoise_bank_key[:kv_compress].copy()
            self.denoise_bank_value = self.denoise_bank_value[:kv_compress].copy()
            for i in range(len(self.denoise_bank_key)):
                self.denoise_bank_key[i] = torch.nan_to_num(self.denoise_bank_key[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)
                self.denoise_bank_value[i] = torch.nan_to_num(self.denoise_bank_value[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)
        if len(self.derain_bank_key) != 0 :
            self.derain_bank_key = self.derain_bank_key[:kv_compress].copy()
            self.derain_bank_value = self.derain_bank_value[:kv_compress].copy()
            for i in range(len(self.derain_bank_key)):
                self.derain_bank_key[i] = torch.nan_to_num(self.derain_bank_key[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)
                self.derain_bank_value[i] = torch.nan_to_num(self.derain_bank_value[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)
        if len(self.dehaze_bank_key) != 0 :
            self.dehaze_bank_key = self.dehaze_bank_key[:kv_compress].copy()
            self.dehaze_bank_value = self.dehaze_bank_value[:kv_compress].copy()
            for i in range(len(self.dehaze_bank_key)):
                self.dehaze_bank_key[i] = torch.nan_to_num(self.dehaze_bank_key[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)
                self.dehaze_bank_value[i] = torch.nan_to_num(self.dehaze_bank_value[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)
        if len(self.deblur_bank_key) != 0 :
            self.deblur_bank_key = self.deblur_bank_key[:kv_compress].copy()
            self.deblur_bank_value = self.deblur_bank_value[:kv_compress].copy()
            for i in range(len(self.deblur_bank_key)):
                self.deblur_bank_key[i] = torch.nan_to_num(self.deblur_bank_key[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)
                self.deblur_bank_value[i] = torch.nan_to_num(self.deblur_bank_value[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)

        if len(self.lowlight_bank_key) != 0:
            self.lowlight_bank_key = self.lowlight_bank_key[:kv_compress].copy()
            self.lowlight_bank_value = self.lowlight_bank_value[:kv_compress].copy()
            for i in range(len(self.lowlight_bank_key)):
                self.lowlight_bank_key[i] = torch.nan_to_num(self.lowlight_bank_key[i].clone().detach(), nan=0.0, posinf=1,
                                                           neginf=-1)
                self.lowlight_bank_value[i] = torch.nan_to_num(self.lowlight_bank_value[i].clone().detach(), nan=0.0, posinf=1,
                                                             neginf=-1)



    def comprehensive_attention_processing(self, mk, qk, mv, n):

        device = mk.device
        batch_size = qk.shape[0]
        n_total = n
        h, w = qk.shape[2], qk.shape[3]
        ck = mk.shape[1]

        mk_flat = mk.view(n_total, -1)  # [n_total, ck*h*w]
        qk_flat = qk.view(batch_size, -1)  # [batch_size, ck*h*w]
        mv_flat = mv.view(n_total, -1)  # [n_total, 512*h*w]
        mk_flat = F.normalize(mk_flat, dim=1)
        qk_flat = F.normalize(qk_flat, dim=1)
        similarity = torch.matmul(qk_flat, mk_flat.t()) / math.sqrt(ck * h * w)
        weights = torch.softmax(similarity, dim=1)
        weighted_output_flat = torch.matmul(weights, mv_flat)
        output = weighted_output_flat.view(batch_size, self.value_dim , h, w)
        return output



    def save_prompts(self, epoch, save_root="./save_prompts"):
        pairs = [("denoise", self.denoise_bank_key, self.denoise_bank_value),
                 ("derain", self.derain_bank_key, self.derain_bank_value),
                 ("dehaze", self.dehaze_bank_key, self.dehaze_bank_value),
                 ("deblur", self.deblur_bank_key, self.deblur_bank_value),
                 ("lowlight", self.lowlight_bank_key, self.lowlight_bank_value),
                 ]

        for i, (deg_type, key, value) in enumerate(pairs):
            pair_dir = os.path.join(save_root, str(epoch))
            if not os.path.exists(pair_dir):
                os.makedirs(pair_dir)

            if deg_type in self.deg_types[0]:
                k = [t.cpu() for t in key]
                v = [t.cpu() for t in value]

                torch.save(k, os.path.join(pair_dir, f"{deg_type}_key.pt"))
                torch.save(v, os.path.join(pair_dir, f"{deg_type}_value.pt"))
    def load_prompts(self, epoch, save_root="./save_prompts", amp=True, drop_last=True, deg_type="denoise"):
        pair_dir = os.path.join(save_root, str(epoch))
        if drop_last:
            kv_compress = self.T_max - self.update_step + 1
        else:
            kv_compress = self.T_max

        if deg_type == "denoise":
            self.denoise_bank_key = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "denoise_key.pt"))[:kv_compress]]
            self.denoise_bank_value = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "denoise_value.pt"))[:kv_compress]]
        elif deg_type == "derain":
            self.derain_bank_key = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "derain_key.pt"))[:kv_compress]]
            self.derain_bank_value = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "derain_value.pt"))[:kv_compress]]
        elif deg_type == "dehaze":
            self.dehaze_bank_key = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "dehaze_key.pt"))[:kv_compress]]
            self.dehaze_bank_value = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "dehaze_value.pt"))[:kv_compress]]
        elif deg_type == "deblur":
            self.deblur_bank_key = [t.cuda().float() if not amp else t.cuda() for t in
                                    torch.load(os.path.join(pair_dir, "deblur_key.pt"))[:kv_compress]]
            self.deblur_bank_value = [t.cuda().float() if not amp else t.cuda() for t in
                                      torch.load(os.path.join(pair_dir, "deblur_value.pt"))[:kv_compress]]
        elif deg_type == "lowlight":
            self.lowlight_bank_key = [t.cuda().float() if not amp else t.cuda() for t in
                                    torch.load(os.path.join(pair_dir, "lowlight_key.pt"))[:kv_compress]]
            self.lowlight_bank_value = [t.cuda().float() if not amp else t.cuda() for t in
                                      torch.load(os.path.join(pair_dir, "lowlight_value.pt"))[:kv_compress]]

        print(f"loaded epoch={str(pair_dir)} prompts")
    def plot_tsne(self, keys, values, ids,
                             n_samples=64, random_state=42, save_path="./t-SNE", epoch=0):
        """Plot side-by-side t-SNE maps for keys and values."""

        def prepare_features(feature_lists, ids):
            all_features = []
            all_labels = []
            for i, feature_list in zip(ids, feature_lists):
                if len(feature_list) > 0:
                    features = torch.cat(feature_list, dim=0)
                    features_flat = features.flatten(start_dim=1)
                    all_features.append(features_flat.cpu().numpy())
                    all_labels.extend([i] * features.shape[0])
            if len(all_features) == 0:
                return None, None
            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.array(all_labels)

            if all_features.shape[0] > n_samples:
                idx = np.random.choice(all_features.shape[0], n_samples, replace=False)
                all_features = all_features[idx]
                all_labels = all_labels[idx]
            return all_features, all_labels

        def run_tsne_and_plot(features, labels, ax, title, colors, ids):
            tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
            features_2d = tsne.fit_transform(features)
            for i, uid in enumerate(ids):
                idx = (labels == uid)
                if np.any(idx):
                    ax.scatter(features_2d[idx, 0], features_2d[idx, 1],
                               c=colors[i % len(colors)], label=f"{uid}", alpha=0.6, s=20)
            ax.set_title(title)
            ax.legend()

        colors = ["red", "blue", "green", "orange", "purple", "cyan", "olive", "navy","teal", "coral"]

        feats_left, labels_left = prepare_features(keys, ids)
        feats_right, labels_right = prepare_features(values, ids)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if feats_left is not None:
            run_tsne_and_plot(feats_left, labels_left, axes[0], "Keys", colors, ids)
        if feats_right is not None:
            run_tsne_and_plot(feats_right, labels_right, axes[1], "Values", colors, ids)

        plt.tight_layout()
        if save_path is not None:
            save_path = os.path.join(save_path, str(epoch))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, "t-SNE.png"), dpi=300, bbox_inches="tight")
            print(f"Saved t-SNE plot to: {save_path}")
        else:
            plt.show()
    def save_tSNE(self, epoch=0):
        kv_compress = self.T_max - self.update_step + 1
        with torch.no_grad():
            keys = [self.denoise_bank_key[:kv_compress], self.derain_bank_key[:kv_compress], self.dehaze_bank_key[:kv_compress], self.deblur_bank_key[:kv_compress], self.lowlight_bank_key[:kv_compress]]
            values = [self.denoise_bank_value[:kv_compress], self.derain_bank_value[:kv_compress], self.dehaze_bank_value[:kv_compress], self.deblur_bank_value[:kv_compress], self.lowlight_bank_value[:kv_compress]]
            self.plot_tsne(keys, values, ids=self.deg_types, epoch=epoch)

