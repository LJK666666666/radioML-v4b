#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PETCGDNN 的 PyTorch 移植版（与 src/model/pet_model.py 的 Keras 架构一一对应）。

用于 RTX 5070 Ti (Blackwell sm_120) 上的 GPU 训练 —— 现有 radioml 环境的 TF 2.13 不支持该卡，
故把参考分类器 / 下游分类器移到 PyTorch + cu128。

架构（对应 PETCGDNN2016.py）：
  输入 (B, 2, L)
  1. 相位估计：flatten(转置后的IQ) -> Linear(2L,1) -> 标量相位 phi
  2. 相位旋转：用 cos(phi)/sin(phi) 对 I/Q 做旋转，得到对齐后的 (I', Q')
  3. Conv2D(75,(8,2),valid,relu) -> Conv2D(25,(5,1),valid,relu)   (输入按 H=L, W=2, C=1)
  4. GRU(128) 取末状态
  5. Linear(num_classes)  (softmax 由 CrossEntropyLoss 内部处理，logits 输出)
"""

import torch
import torch.nn as nn


class PETCGDNN(nn.Module):
    def __init__(self, input_shape=(2, 128), num_classes=11):
        super().__init__()
        self.seq_len = int(input_shape[1])
        L = self.seq_len

        # 1. 相位估计：对 flatten 的 (L,2) = 2L 维做线性投影到 1 维角度
        self.phase_fc = nn.Linear(2 * L, 1)

        # 3. 空间特征提取（Keras Conv2D 通道在末，这里用 PyTorch (B,C,H,W)，H=L, W=2）
        self.conv1 = nn.Conv2d(1, 75, kernel_size=(8, 2))   # valid -> (B,75,L-7,1)
        self.conv2 = nn.Conv2d(75, 25, kernel_size=(5, 1))  # valid -> (B,25,L-11,1)
        self.relu = nn.ReLU()

        self.temporal_dim = L - 11

        # 4. 时序特征：GRU(128)，batch_first
        self.gru = nn.GRU(input_size=25, hidden_size=128, batch_first=True)

        # 5. 分类头
        self.fc = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 2, L)
        B, _, L = x.shape
        i = x[:, 0, :]   # (B, L)
        q = x[:, 1, :]   # (B, L)

        # 1. 相位估计：转置成 (B,L,2) 再 flatten 成 (B,2L)，顺序 [I_0,Q_0,I_1,Q_1,...]
        main_flat = x.permute(0, 2, 1).reshape(B, -1)   # (B, 2L)
        phi = self.phase_fc(main_flat)                  # (B, 1)
        cos1 = torch.cos(phi)                           # (B, 1)
        sin1 = torch.sin(phi)

        # 2. 相位旋转
        y1 = i * cos1 + q * sin1   # (B, L) 旋转后 I
        y2 = q * cos1 - i * sin1   # (B, L) 旋转后 Q

        # 拼成 (B, L, 2) -> (B, 1, L, 2)  对应 Keras (H=L, W=2, C=1)
        xr = torch.stack([y1, y2], dim=2)              # (B, L, 2)
        xr = xr.unsqueeze(1)                           # (B, 1, L, 2)

        # 3. 卷积
        xr = self.relu(self.conv1(xr))                 # (B,75,L-7,1)
        xr = self.relu(self.conv2(xr))                 # (B,25,L-11,1)
        xr = xr.squeeze(-1)                            # (B,25,L-11)
        xr = xr.permute(0, 2, 1)                       # (B, L-11, 25)

        # 4. GRU
        out, h = self.gru(xr)                          # h: (1, B, 128)
        feat = h[-1]                                   # (B, 128)

        # 5. 分类（输出 logits）
        return self.fc(feat)


def build_pet_torch(input_shape=(2, 128), num_classes=11):
    return PETCGDNN(input_shape=input_shape, num_classes=num_classes)
