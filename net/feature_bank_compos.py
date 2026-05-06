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
                 dehaze_bank_key=None,
                 dehaze_bank_value=None,
                 derain_bank_key=None,
                 derain_bank_value=None,
                 desnow_bank_key=None,
                 desnow_bank_value=None,
                 lowlight_bank_key=None,
                 lowlight_bank_value=None,
                 ):
        super().__init__()
        self.deg_types = ["dehaze", "derain", "desnow", "lowlight"]
        self.T_max = T_max
        self.update_step = T_max//5
        self.key_dim=key_dim
        self.value_dim=value_dim

        self.update_dehaze_key_3Dconv = nn.Sequential(
            nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(self.update_step, 3, 3), padding=(0, 1, 1), groups=key_dim),
            nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(key_dim),
            nn.ReLU(),

            )
        self.update_dehaze_value_3Dconv = nn.Sequential(
            nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(self.update_step, 3, 3), padding=(0, 1, 1), groups=value_dim),
            nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(value_dim),
            nn.ReLU(),
        )

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

        self.update_desnow_key_3Dconv = nn.Sequential(
            nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(self.update_step, 3, 3),
                      padding=(0, 1, 1), groups=key_dim),
            nn.Conv3d(in_channels=key_dim, out_channels=key_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(key_dim),
            nn.ReLU(),
        )
        self.update_desnow_value_3Dconv = nn.Sequential(
            nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(self.update_step, 3, 3),
                      padding=(0, 1, 1), groups=value_dim),
            nn.Conv3d(in_channels=value_dim, out_channels=value_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(value_dim),
            nn.ReLU(),
        )


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

        self.init_bank(dehaze_bank_key, dehaze_bank_value, derain_bank_key, derain_bank_value, desnow_bank_key, desnow_bank_value, lowlight_bank_key, lowlight_bank_value)

    def init_bank(self,
                  dehaze_bank_key=None,
                  dehaze_bank_value=None,
                  derain_bank_key=None,
                  derain_bank_value=None,
                  desnow_bank_key=None,
                  desnow_bank_value=None,
                  lowlight_bank_key=None,
                  lowlight_bank_value=None):
        self.dehaze_bank_key = dehaze_bank_key if dehaze_bank_key is not None else []
        self.dehaze_bank_value = dehaze_bank_value if dehaze_bank_value is not None else []

        self.derain_bank_key = derain_bank_key if derain_bank_key is not None else []
        self.derain_bank_value = derain_bank_value if derain_bank_value is not None else []

        self.desnow_bank_key = desnow_bank_key if desnow_bank_key is not None else []
        self.desnow_bank_value = desnow_bank_value if desnow_bank_value is not None else []

        self.lowlight_bank_key = lowlight_bank_key if lowlight_bank_key is not None else []
        self.lowlight_bank_value = lowlight_bank_value if lowlight_bank_value is not None else []




    def update_bank(self, deg_type, mk, mv):
        if deg_type == "dehaze":
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
                    self.dehaze_bank_key.append(mk[i:i + 1])
                    self.dehaze_bank_value.append(mv[i:i + 1])

                else:
                    indices = list(range(len(self.dehaze_bank_key)))
                    random.shuffle(indices)
                    shuffled_key = [self.dehaze_bank_key[j] for j in indices]
                    shuffled_value = [self.dehaze_bank_value[j] for j in indices]
                    key_combined = torch.cat(shuffled_key, dim=0).unsqueeze(0).permute(0, 2, 1, 3,
                                                                                       4)  # [1, 64, T_max, h, w]
                    value_combined = torch.cat(shuffled_value, dim=0).unsqueeze(0).permute(0, 2, 1, 3,
                                                                                           4)  # [1, 512, T_max,  h, w]
                    self.dehaze_bank_key.clear()
                    self.dehaze_bank_value.clear()
                    update_keys = self.update_dehaze_key_3Dconv(key_combined).squeeze(0).permute(1, 0, 2,
                                                                                                 3)  # # [1, 64, update_step, h, w]
                    update_values = self.update_dehaze_value_3Dconv(value_combined).squeeze(0).permute(1, 0, 2,
                                                                                                       3)  # # [1, 64, update_step, h, w]
                    self.dehaze_bank_key = [update_keys[i:i + 1] for i in range(update_keys.shape[0])]
                    self.dehaze_bank_value = [update_values[i:i + 1] for i in range(update_keys.shape[0])]
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
        elif deg_type == "desnow":
            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            k_list = [mk for _ in range(n)]
            mk = torch.cat(k_list, dim=0)  # [B*n, 512, H, W]

            v_list = [mv for _ in range(n)]
            mv = torch.cat(v_list, dim=0)  # [B*n, 512, H, W]
            for i in range(mk.shape[0]):
                if len(self.desnow_bank_key) < self.T_max:
                    self.desnow_bank_key.append(mk[i:i+1])
                    self.desnow_bank_value.append(mv[i:i+1])

                else:
                    indices = list(range(len(self.desnow_bank_key)))
                    random.shuffle(indices)
                    shuffled_key = [self.desnow_bank_key[j] for j in indices]
                    shuffled_value = [self.desnow_bank_value[j] for j in indices]
                    key_combined = torch.cat(shuffled_key, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 64, T_max, h, w]
                    value_combined = torch.cat(shuffled_value, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 512, T_max,  h, w]
                    self.desnow_bank_key.clear()
                    self.desnow_bank_value.clear()
                    update_keys = self.update_desnow_key_3Dconv(key_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    update_values = self.update_desnow_value_3Dconv(value_combined).squeeze(0).permute(1, 0, 2, 3) # # [1, 64, update_step, h, w]
                    self.desnow_bank_key = [update_keys[i:i+1] for i in range(update_keys.shape[0])]
                    self.desnow_bank_value = [update_values[i:i+1] for i in range(update_keys.shape[0])]
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

        if  len(self.derain_bank_key) == 0 or len(self.dehaze_bank_key) == 0 or len(self.desnow_bank_key) == 0 or len(self.lowlight_bank_key) == 0:
            return None, 0
        mk = torch.cat([torch.cat(self.dehaze_bank_key, dim=0), torch.cat(self.derain_bank_key, dim=0), torch.cat(self.desnow_bank_key, dim=0), torch.cat(self.lowlight_bank_key, dim=0)], dim=0)

        mv = torch.cat([torch.cat(self.dehaze_bank_value, dim=0), torch.cat(self.derain_bank_value, dim=0), torch.cat(self.desnow_bank_value, dim=0), torch.cat(self.lowlight_bank_value, dim=0)], dim=0)

        n1 = len(self.dehaze_bank_key)
        n2 = len(self.derain_bank_key)
        n3 = len(self.desnow_bank_key)
        n4 = len(self.lowlight_bank_key)


        label, readout = self.comprehensive_attention_processing(mk, qk, mv, n1, n2, n3, n4, interact_label=interact_label)


        return label, readout

    def clear_grad(self, stage=0):
        if stage == 0:

            self.dehaze_bank_key.clear()
            self.derain_bank_key.clear()
            self.desnow_bank_key.clear()
            self.lowlight_bank_key.clear()

            self.dehaze_bank_value.clear()
            self.desnow_bank_value.clear()
            self.derain_bank_value.clear()
            self.lowlight_bank_value.clear()

            return
        kv_compress = self.T_max - self.update_step + 1
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
        if len(self.desnow_bank_key) != 0 :
            self.desnow_bank_key = self.desnow_bank_key[:kv_compress].copy()
            self.desnow_bank_value = self.desnow_bank_value[:kv_compress].copy()
            for i in range(len(self.desnow_bank_key)):
                self.desnow_bank_key[i] = torch.nan_to_num(self.desnow_bank_key[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)
                self.desnow_bank_value[i] = torch.nan_to_num(self.desnow_bank_value[i].clone().detach(), nan=0.0, posinf=1, neginf=-1)

        if len(self.lowlight_bank_key) != 0:
            self.lowlight_bank_key = self.lowlight_bank_key[:kv_compress].copy()
            self.lowlight_bank_value = self.lowlight_bank_value[:kv_compress].copy()
            for i in range(len(self.lowlight_bank_key)):
                self.lowlight_bank_key[i] = torch.nan_to_num(self.lowlight_bank_key[i].clone().detach(), nan=0.0, posinf=1,
                                                           neginf=-1)
                self.lowlight_bank_value[i] = torch.nan_to_num(self.lowlight_bank_value[i].clone().detach(), nan=0.0, posinf=1,
                                                             neginf=-1)



    def comprehensive_attention_processing(self,
            mk, qk, mv, n1, n2, n3, n4,
            pred_method: Literal['max', 'mean', 'weighted', 'topk', 'max_abs'] = 'topk',
            topk_k: int = 3,
            weighted_alpha: float = 0.7,
            interact_label=None
    ):
        """Class-aware memory attention for compositional degradations."""
        device = mk.device
        batch_size = qk.shape[0]
        n_total = n1 + n2 + n3 + n4
        h, w = qk.shape[2], qk.shape[3]
        ck = mk.shape[1]

        mk_flat = mk.view(n_total, -1)  # [n_total, ck*h*w]
        qk_flat = qk.view(batch_size, -1)  # [batch_size, ck*h*w]
        mv_flat = mv.view(n_total, -1)  # [n_total, 512*h*w]
        mk_flat = F.normalize(mk_flat, dim=1)
        qk_flat = F.normalize(qk_flat, dim=1)
        similarity = torch.matmul(qk_flat, mk_flat.t()) / math.sqrt(ck * h * w)

        def get_class_scores(method):
            """Compute class scores with the selected reduction rule."""
            if method == 'max':
                return [
                    similarity[:, :n1].max(dim=1).values,
                    similarity[:, n1:n1 + n2].max(dim=1).values,
                    similarity[:, n1 + n2:n1+n2+n3].max(dim=1).values,
                    similarity[:, n1 + n2+n3:].max(dim=1).values,

                ]

            elif method == 'mean':
                return [
                    similarity[:, :n1].mean(dim=1),
                    similarity[:, n1:n1 + n2].mean(dim=1),
                    similarity[:, n1 + n2:n1+n2+n3].mean(dim=1),
                    similarity[:, n1 + n2+n3:].mean(dim=1),

                ]

            elif method == 'topk':
                def topk_mean(sims, k):
                    if k == 0:
                        return torch.full((batch_size,), float('-inf'), device=device)
                    topk_vals = torch.topk(sims, k, dim=1).values
                    return topk_vals.mean(dim=1)
                topk_k = min(n1, n2, n3, n4)
                return [
                    topk_mean(similarity[:, :n1], topk_k),
                    topk_mean(similarity[:, n1:n1 + n2], topk_k),
                    topk_mean(similarity[:, n1 + n2:n1+n2+n3], topk_k),
                    topk_mean(similarity[:, n1 + n2+n3:], topk_k),

                ]

        scores_n1, scores_n2, scores_n3, scores_n4 = get_class_scores(pred_method)
        scores = torch.stack([scores_n1, scores_n2, scores_n3, scores_n4], dim=1)  # [batch_size, 4]
        probs = torch.sigmoid(scores)
        predictions = (probs > 0.5).int()

        if interact_label is not None:
            predictions = interact_label

        class_masks = torch.zeros(4, n_total, dtype=torch.bool, device=device)
        class_masks[0, :n1] = True
        class_masks[1, n1:n1 + n2] = True
        class_masks[2, n1 + n2:n1+n2+n3] = True
        class_masks[3, n1 + n2+n3:] = True

        B = predictions.shape[0]

        predictions_bool = predictions.bool()
        class_masks_expanded = class_masks.unsqueeze(0).expand(B, -1, -1)
        pred_expanded = predictions_bool.unsqueeze(-1)
        sample_masks = (class_masks_expanded & pred_expanded).any(dim=1)
        masked_similarity = similarity.clone()
        masked_similarity[~sample_masks] = -1e6

        weights = torch.softmax(masked_similarity, dim=1)

        weighted_output_flat = torch.matmul(weights, mv_flat)
        output = weighted_output_flat.view(batch_size, self.value_dim , h, w)
        return scores, output



    def save_prompts(self, epoch, save_root="./save_prompts"):
        pairs = [("dehaze", self.dehaze_bank_key, self.dehaze_bank_value),
                 ("derain", self.derain_bank_key, self.derain_bank_value),
                 ("desnow", self.desnow_bank_key, self.desnow_bank_value),
                 ("lowlight", self.lowlight_bank_key, self.lowlight_bank_value),
                 ]

        for i, (deg_type, key, value) in enumerate(pairs):
            pair_dir = os.path.join(save_root, str(epoch))
            if not os.path.exists(pair_dir):
                os.makedirs(pair_dir)

            k = [t.cpu() for t in key]
            v = [t.cpu() for t in value]

            torch.save(k, os.path.join(pair_dir, f"{deg_type}_key.pt"))
            torch.save(v, os.path.join(pair_dir, f"{deg_type}_value.pt"))
    def load_prompts(self, prompts_name, save_root="./save_prompts", amp=True, drop_last=True):
        pair_dir = os.path.join(save_root, str(prompts_name))
        if drop_last:
            kv_compress = self.T_max - self.update_step + 1
        else:
            kv_compress = self.T_max
        self.dehaze_bank_key = [t.cuda().float() if not amp else t.cuda() for t in
                                torch.load(os.path.join(pair_dir, "dehaze_key.pt"))[:kv_compress]]
        self.dehaze_bank_value = [t.cuda().float() if not amp else t.cuda() for t in
                                  torch.load(os.path.join(pair_dir, "dehaze_value.pt"))[:kv_compress]]
        self.derain_bank_key = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "derain_key.pt"))[:kv_compress]]
        self.derain_bank_value = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "derain_value.pt"))[:kv_compress]]
        self.desnow_bank_key = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "desnow_key.pt"))[:kv_compress]]
        self.desnow_bank_value = [t.cuda().float() if not amp else t.cuda() for t in torch.load(os.path.join(pair_dir, "desnow_value.pt"))[:kv_compress]]

        self.lowlight_bank_key = [t.cuda().float() if not amp else t.cuda() for t in
                                torch.load(os.path.join(pair_dir, "lowlight_key.pt"))[:kv_compress]]
        self.lowlight_bank_value = [t.cuda().float() if not amp else t.cuda() for t in
                                  torch.load(os.path.join(pair_dir, "lowlight_value.pt"))[:kv_compress]]

        print(f"loaded {str(pair_dir)} prompts")
    def plot_tsne(self, keys, values, ids,
                             n_samples=64, random_state=42, save_path="./t-SNE", path=""):

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
            save_path = os.path.join(save_path, str(path))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, "t-SNE.png"), dpi=300, bbox_inches="tight")
            print(f"Saved t-SNE plot to: {save_path}")
        else:
            plt.show()
    def save_tSNE(self, path=""):
        kv_compress = self.T_max - self.update_step + 1
        with torch.no_grad():
            keys = [self.dehaze_bank_key[:kv_compress], self.derain_bank_key[:kv_compress], self.desnow_bank_key[:kv_compress], self.lowlight_bank_key[:kv_compress]]
            values = [self.dehaze_bank_value[:kv_compress], self.derain_bank_value[:kv_compress], self.desnow_bank_value[:kv_compress], self.lowlight_bank_value[:kv_compress]]
            self.plot_tsne(keys, values, ids=self.deg_types, path=path)

