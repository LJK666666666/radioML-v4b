#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学习式残差去噪器(1D Res-UNet),用于替代 GPR 做 I/Q 去噪。

设计目标:输出仍是 (B,2,128) → 对所有分类器 drop-in 替换 GPR;非线性、学调制结构,
有望在低过采样(spS=2)上超越线性平滑。

- 预测噪声残差 N̂,去噪结果 = x - N̂(残差学习,易优化、近似恒等更稳)。
- 小卷积核(k=3)避免跨符号过平滑(GPR 在 spS=2 的失败模式)。
- 1D U-Net + skip connection,沿时间维 128→64→32 两次下采样。
- blind 去噪(不需要 SNR 输入):训练时对高SNR纯净样本加多种强度合成噪声,学会盲去噪。
"""

import torch
import torch.nn as nn


def _block(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=k // 2),
        nn.BatchNorm1d(cout),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv1d(cout, cout, k, padding=k // 2),
        nn.BatchNorm1d(cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ResidualDenoiser1D(nn.Module):
    def __init__(self, ch=2, base=32):
        super().__init__()
        self.enc1 = _block(ch, base)            # (B,base,128)
        self.down1 = nn.Conv1d(base, base, 2, stride=2)        # 128->64
        self.enc2 = _block(base, base * 2)      # (B,2base,64)
        self.down2 = nn.Conv1d(base * 2, base * 2, 2, stride=2)  # 64->32
        self.bott = _block(base * 2, base * 4)  # (B,4base,32)
        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, 2, stride=2)  # 32->64
        self.dec2 = _block(base * 4, base * 2)  # cat skip -> (B,2base,64)
        self.up1 = nn.ConvTranspose1d(base * 2, base, 2, stride=2)      # 64->128
        self.dec1 = _block(base * 2, base)      # cat skip -> (B,base,128)
        self.out = nn.Conv1d(base, ch, 1)       # 预测残差 (B,2,128)
        # 残差头零初始化:初始 ≈ 恒等(不去噪),训练从"无害"开始
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bott(self.down2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        noise = self.out(d1)
        return x - noise        # 去噪结果


def build_residual_denoiser(ch=2, base=32):
    return ResidualDenoiser1D(ch=ch, base=base)
