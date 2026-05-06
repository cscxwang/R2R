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
    对 5D 特征进行 LayerNorm。
    输入: [1, C, N, H, W] (1 不是 batch)
    可以指定归一化维度: 'C' 或 'N'
    """
    def __init__(self, normalized_dim, num_features, eps: float = 1e-6):
        """
        Args:
            normalized_dim: str, 'C' 或 'N'
            num_features: int, 对应维度的大小
            eps: float, 防止除零
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
            # 对 C 维做 LayerNorm
            # 先 permute 到 [N,H,W,C]
            x_perm = x.permute(1,2,3,0)  # [N,H,W,C]
            x_norm = self.layer_norm(x_perm)
            # permute 回原始顺序
            x_out = x_norm.permute(3,0,1,2)  # [1,C,N,H,W]
        else:
            # 对 N 维做 LayerNorm
            # 先 permute 到 [C,H,W,N]
            x_perm = x.permute(0,2,3,1)  # [C,H,W,N]
            x_norm = self.layer_norm(x_perm)
            # permute 回原始顺序
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
        self.deg_types = ["dehaze", "derain", "desnow", "lowlight"]  # 退化类型
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
        # mk = mk.clone().detach()
        # mv = mv.clone().detach()

        if deg_type == "dehaze":
            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            # n  # 拼接倍数

            # 先复制成列表，再拼接
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
            # n  # 拼接倍数

            # 先复制成列表，再拼接
            k_list = [mk for _ in range(n)]
            mk = torch.cat(k_list, dim=0)  # [B*n, 512, H, W]

            v_list = [mv for _ in range(n)]
            mv = torch.cat(v_list, dim=0)  # [B*n, 512, H, W]
            for i in range(mk.shape[0]):
                if len(self.derain_bank_key) < self.T_max:
                    self.derain_bank_key.append(mk[i:i+1])
                    self.derain_bank_value.append(mv[i:i+1])
                    # return

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
                    # self.derain_bank_key.append(mk[i:i + 1])
                    # self.derain_bank_value.append(mv[i:i + 1])


        elif deg_type == "desnow":
            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            # n  # 拼接倍数

            # 先复制成列表，再拼接
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
                    #
                    # self.dehaze_bank_key.append(mk[i:i + 1])
                    # self.dehaze_bank_value.append(mv[i:i + 1])

        elif deg_type == "lowlight":
            B, C, H, W = mk.shape
            if B == 0:
                return
            n = self.T_max // B + 1
            # n  # 拼接倍数

            # 先复制成列表，再拼接
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
        # kv_compress = 10000000
        # print(len(self.derain_bank_key))

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
        """
           数值稳定的高效处理版本

           Args:
               mk: [n1+n2+n3, 64, h, w] 记忆键
               qk: [16, 64, h, w] 查询键
               mv: [n1+n2+n3, 512, h, w] 记忆值
               n1, n2, n3: 三个类别的样本数量

           Returns:
               predictions: [16] 预测类别
               output: [16, 512, h, w] 输出
               similarity: [16, n1+n2+n3] 相似度矩阵
           """
        """
        完整的多策略注意力处理版本

        Args:
            pred_method: 类别预测方法
                'max' - 使用原始最大值（推荐）
                'mean' - 使用平均值
                'weighted' - 加权组合最大值和平均值
                'topk' - 使用top-k平均值
                'max_abs' - 使用绝对值最大值（原方案）
            topk_k: top-k方法中的k值
            weighted_alpha: 加权方法中最大值的权重
        """
        device = mk.device
        batch_size = qk.shape[0]
        n_total = n1 + n2 + n3 + n4
        h, w = qk.shape[2], qk.shape[3]
        ck = mk.shape[1]

        # 步骤1: 展平空间维度
        mk_flat = mk.view(n_total, -1)  # [n_total, ck*h*w]
        # if torch.isnan(mk_flat).any() or torch.isinf(mk_flat).any():
        #     print("mk_flat is nan")
        #     assert 1==0
        #
        #     print(n1, n2, n3)
        qk_flat = qk.view(batch_size, -1)  # [batch_size, ck*h*w]
        # if torch.isnan(qk_flat).any() or torch.isinf(qk_flat).any():
        #     print("qk_flat is nan")
        #     assert 1==0


        mv_flat = mv.view(n_total, -1)  # [n_total, 512*h*w]
        # if torch.isnan(mv_flat).any() or torch.isinf(mv_flat).any():
        #     print("mv_flat is nan")
        #     assert 1==0

        # 步骤2: 计算稳定的相似度

        # 步骤1.5: L2归一化
        mk_flat = F.normalize(mk_flat, dim=1)  # 每行向量单位化
        qk_flat = F.normalize(qk_flat, dim=1)

        # 步骤2: 计算稳定的相似度
        # 使用归一化向量后的dot product，相当于 cosine similarity
        similarity = torch.matmul(qk_flat, mk_flat.t()) / math.sqrt(ck * h * w)


        # mk_norm_sq = torch.sum(mk_flat ** 2, dim=1)  # [n_total]
        # # mk_norm_sq = torch.clamp(mk_norm_sq, max=1e3)  # 或更合理的阈值
        # dot_product = torch.matmul(qk_flat, mk_flat.t())  # [batch_size, n_total]
        # similarity = (dot_product -  0.5 * mk_norm_sq.unsqueeze(0)) / math.sqrt(ck * h * w)

        # 步骤3: 多种类别预测策略
        def get_class_scores(method):
            """根据不同方法计算类别得分"""
            if method == 'max':
                # 原始最大值
                return [
                    similarity[:, :n1].max(dim=1).values,
                    similarity[:, n1:n1 + n2].max(dim=1).values,
                    similarity[:, n1 + n2:n1+n2+n3].max(dim=1).values,
                    similarity[:, n1 + n2+n3:].max(dim=1).values,

                ]

            elif method == 'mean':
                # 平均值
                return [
                    similarity[:, :n1].mean(dim=1),
                    similarity[:, n1:n1 + n2].mean(dim=1),
                    similarity[:, n1 + n2:n1+n2+n3].mean(dim=1),
                    similarity[:, n1 + n2+n3:].mean(dim=1),

                ]

            # elif method == 'weighted':
            #     # 加权组合
            #     max_n1 = similarity[:, :n1].max(dim=1).values
            #     mean_n1 = similarity[:, :n1].mean(dim=1)
            #     score_n1 = weighted_alpha * max_n1 + (1 - weighted_alpha) * mean_n1
            #
            #     max_n2 = similarity[:, n1:n1 + n2].max(dim=1).values
            #     mean_n2 = similarity[:, n1:n1 + n2].mean(dim=1)
            #     score_n2 = weighted_alpha * max_n2 + (1 - weighted_alpha) * mean_n2
            #
            #     max_n3 = similarity[:, n1 + n2:].max(dim=1).values
            #     mean_n3 = similarity[:, n1 + n2:].mean(dim=1)
            #     score_n3 = weighted_alpha * max_n3 + (1 - weighted_alpha) * mean_n3
            #
            #     return [score_n1, score_n2, score_n3]

            elif method == 'topk':
                # top-k平均值
                def topk_mean(sims, k):
                    # k = min(k, sims.shape[1])
                    # print(k)
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

            # elif method == 'max_abs':
            #     # 绝对值最大值（原方案）
            #     abs_sim = torch.abs(similarity)
            #     return [
            #         abs_sim[:, :n1].max(dim=1).values,
            #         abs_sim[:, n1:n1 + n2].max(dim=1).values,
            #         abs_sim[:, n1 + n2:].max(dim=1).values
            #     ]

        # 计算类别得分
        scores_n1, scores_n2, scores_n3, scores_n4 = get_class_scores(pred_method)
        scores = torch.stack([scores_n1, scores_n2, scores_n3, scores_n4], dim=1)  # [batch_size, 4]

        probs = torch.sigmoid(scores)  # [B, 4] 每一维是属于该类的概率
        predictions = (probs > 0.5).int()



        # predictions = torch.argmax(scores, dim=1)  # [batch_size]

        if interact_label is not None:
            predictions = interact_label


        # if interact_label is  None:
        #     print(predictions)
        # print(scores)
        # print(predictions)
        # print(scores.shape)
        # print(predictions.shape)
        # assert 1 == 0

        # 步骤4: 创建类别掩码
        class_masks = torch.zeros(4, n_total, dtype=torch.bool, device=device)
        class_masks[0, :n1] = True
        class_masks[1, n1:n1 + n2] = True
        class_masks[2, n1 + n2:n1+n2+n3] = True
        class_masks[3, n1 + n2+n3:] = True

        B = predictions.shape[0]

        predictions_bool = predictions.bool()  # [B, 4]

        class_masks_expanded = class_masks.unsqueeze(0).expand(B, -1, -1)  # [B, 4, n_total]
        pred_expanded = predictions_bool.unsqueeze(-1)  # [B, 4, 1]

        # 对每个样本：所有预测为1的类对应的库条目的并集
        sample_masks = (class_masks_expanded & pred_expanded).any(dim=1)  # [B, n_total]



        # 步骤5: 数值稳定的softmax
        masked_similarity = similarity.clone()
        # masked_similarity[~sample_masks] = float('-inf')
        masked_similarity[~sample_masks] = -1e6

        weights = torch.softmax(masked_similarity, dim=1)

        # max_vals = torch.max(masked_similarity, dim=1, keepdim=True)[0]
        # stable_exp = torch.exp(masked_similarity - max_vals)
        # weights = stable_exp / (torch.sum(stable_exp, dim=1, keepdim=True))

        # 步骤6: 加权求和
        weighted_output_flat = torch.matmul(weights, mv_flat)
        output = weighted_output_flat.view(batch_size, self.value_dim , h, w)

        # return predictions, output, similarity, weights
        return scores, output



    def save_prompts(self, epoch, save_root="./save_prompts"):
        # os.makedirs(save_root, exist_ok=True)
        # self.clear_grad()
        pairs = [("dehaze", self.dehaze_bank_key, self.dehaze_bank_value),
                 ("derain", self.derain_bank_key, self.derain_bank_value),
                 ("desnow", self.desnow_bank_key, self.desnow_bank_value),
                 ("lowlight", self.lowlight_bank_key, self.lowlight_bank_value),
                 ]

        for i, (deg_type, key, value) in enumerate(pairs):
            pair_dir = os.path.join(save_root, str(epoch))
            if not os.path.exists(pair_dir):
                os.makedirs(pair_dir)

            # 转 CPU，避免 pickle CUDA tensor 报错
            k = [t.cpu() for t in key]
            v = [t.cpu() for t in value]

            torch.save(k, os.path.join(pair_dir, f"{deg_type}_key.pt"))
            torch.save(v, os.path.join(pair_dir, f"{deg_type}_value.pt"))
        # print(f"saved epoch={str(epoch)} prompts")

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
                    all_labels.extend([i] * features.shape[0])  # 字符串 id
            if len(all_features) == 0:
                return None, None
            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.array(all_labels)

            # 随机采样
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

        # 颜色列表，保证左右一致
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "olive", "navy","teal", "coral"]

        feats_left, labels_left = prepare_features(keys, ids)
        feats_right, labels_right = prepare_features(values, ids)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if feats_left is not None:
            run_tsne_and_plot(feats_left, labels_left, axes[0], "Keys", colors, ids)
        if feats_right is not None:
            run_tsne_and_plot(feats_right, labels_right, axes[1], "Values", colors, ids)

        plt.tight_layout()
        # 保存或显示
        if save_path is not None:
            save_path = os.path.join(save_path, str(path))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, "t-SNE.png"), dpi=300, bbox_inches="tight")
            print(f"t-SNE 图已保存到: {save_path}")
        else:
            plt.show()




    def save_tSNE(self, path=""):
        # self.clear_grad()
        kv_compress = self.T_max - self.update_step + 1
        with torch.no_grad():
            keys = [self.dehaze_bank_key[:kv_compress], self.derain_bank_key[:kv_compress], self.desnow_bank_key[:kv_compress], self.lowlight_bank_key[:kv_compress]]
            values = [self.dehaze_bank_value[:kv_compress], self.derain_bank_value[:kv_compress], self.desnow_bank_value[:kv_compress], self.lowlight_bank_value[:kv_compress]]
            self.plot_tsne(keys, values, ids=self.deg_types, path=path)

