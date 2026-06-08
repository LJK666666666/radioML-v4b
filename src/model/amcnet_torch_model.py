#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AMC-Net 的 PyTorch 移植版（忠实复现原始 PyTorch 源码 AMC-Net/models/model.py，
也与 src/model/amcnet_model.py 的 Keras 架构一一对应）。

用于在新 GPU（PyTorch cu128，如 Blackwell sm_120）上训练 AMC-Net —— 现有 radioml 环境
的 TF 2.13 不支持该卡，故把 AMC-Net 移到 PyTorch + cu128，方便测试去噪预处理方法的效果。

参考论文：AMC-Net: An Effective Network for Automatic Modulation Classification
         (Zhang et al., ICASSP 2023)

架构（默认 sig_len=128, extend_channel=36, conv_chan_list=[36,64,128,256],
      num_heads=2, latent_dim=512=num_heads*256）：
  输入 (B, 2, L)，channel 0 = I，channel 1 = Q
  1. unsqueeze(1) -> (B, 1, 2, L)   把 I/Q 放在 H=2 上，输入通道 C=1
  2. ACM 自适应校正模块：沿 W 维 FFT -> 两个 TinyMLP 分别校正实/虚部 -> 复数乘法 ->
     iFFT 取实部 -> 残差相加。输出 (B, 1, 2, L)
  3. L2 归一化（沿 W 维 dim=-1）
  4. MSM 多尺度模块：3 条并行 Conv2d(1, out//3, kernel=(2,k)) (k=3,5,7)，H=2 被卷成 1，
     沿通道拼接 -> (B, 36, 1, L)
  5. Conv_stem：3 个 Conv_Block（ZeroPad2d((1,1,0,0)) + Conv2d(_,_,(1,3)) + ReLU + BN），
     通道 36->64->128->256，W 不变 -> (B, 256, 1, L)
  6. squeeze(2) -> (B, 256, L)，喂入 FFM
  7. FFM 特征融合模块：对最后一维 W 做 Linear(L->L)，跨通道维 C=256 做多头注意力，
     输出 reshape 成 (B, heads*256=512, head_size=L/heads)
  8. GAP（AdaptiveAvgPool1d(1)）-> (B, 512, 1) -> squeeze(2) -> (B, 512)
  9. classifier：Linear(512,512) + Dropout(0.5) + PReLU + Linear(512, num_classes)
     输出 logits（softmax 由 CrossEntropyLoss 内部处理）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Block(nn.Module):
    """
    对应原版 Conv_Block：
    ZeroPad2d((1, 1, 0, 0)) + Conv2d(in_c, out_c, kernel_size=(1, 3)) + ReLU + BatchNorm2d
    左右各补 1，配合 (1,3) 卷积使宽度 W 不变。
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_c = in_channel
        self.out_c = out_channel

        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(self.in_c, self.out_c, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        return self.conv_block(x)


class MultiScaleModule(nn.Module):
    """
    对应原版 MultiScaleModule：3 条并行多尺度卷积路径。
    每条：ZeroPad2d 左右补 (k-1)/2 + Conv2d(1, out_c//3, kernel=(2, k)) + ReLU + BN。
    kernel 高度=2 把 I/Q 两行卷成 1 行；宽度方向 padding 保持 W 不变。
    输入 (B, 1, 2, W)，输出 (B, out_c, 1, W)。
    """

    def __init__(self, out_channel):
        super().__init__()
        self.out_c = out_channel

        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )
        self.conv_5 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )
        self.conv_7 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 7)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )

    def forward(self, x):
        y1 = self.conv_3(x)
        y2 = self.conv_5(x)
        y3 = self.conv_7(x)
        # 沿通道维拼接 (B, out_c, 1, W)
        return torch.cat([y1, y2, y3], dim=1)


class TinyMLP(nn.Module):
    """
    对应原版 TinyMLP：Linear(N, N//4) + ReLU + Linear(N//4, N) + Tanh。
    作用于最后一维（频域 W=N）。
    """

    def __init__(self, N):
        super().__init__()
        self.N = N
        self.mlp = nn.Sequential(
            nn.Linear(self.N, self.N // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.N // 4, self.N),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class AdaCorrModule(nn.Module):
    """
    对应原版 AdaCorrModule（自适应校正模块，ACM）：
    沿宽度维（dim=-1）做 FFT，得到频谱实部 X_re / 虚部 X_im；
    分别用 TinyMLP 生成校正系数 h_re / h_im，对实/虚部逐元素加权得到校正后的复频谱；
    iFFT 取实部回到时域，最后与输入残差相加。

    在 PyTorch 中用 torch.fft 直接复现频域操作（Keras 版用 tf.signal.fft 实现等价）。
    输入/输出形状均为 (B, C, H, W)，FFT 作用在 W 上。
    """

    def __init__(self, N):
        super().__init__()
        self.N = N
        self.Im = TinyMLP(N)
        self.Re = TinyMLP(N)

    def forward(self, x):
        # x: (B, C, H, W)
        # 残差分支直接引用输入（原版用 copy.deepcopy，对带梯度张量不合适；
        # 这里 x_init 保持为同一引用即可，自适应校正分支会重新构造新张量）。
        x_init = x

        # 沿宽度维做 FFT
        x = torch.fft.fft(x, dim=-1)
        X_re = torch.real(x)
        X_im = torch.imag(x)

        # 自适应校正系数（实/虚部各一支 TinyMLP，作用于最后一维 W）
        h_re = self.Re(X_re)
        h_im = self.Im(X_im)

        # 复数乘法：实部 h_re*X_re，虚部 h_im*X_im
        x = torch.mul(h_re, X_re) + 1j * torch.mul(h_im, X_im)

        # iFFT 取实部回到时域
        x = torch.real(torch.fft.ifft(x, dim=-1))

        # 残差连接
        x = x + x_init
        return x


class FeaFusionModule(nn.Module):
    """
    对应原版 FeaFusionModule（特征融合模块，FFM）：
    输入 (B, C, W)，把通道维 C 当作 token 序列，对最后一维 W 做 query/key/value 线性投影，
    在通道维之间做多头自注意力。
    输出 reshape 成 (B, num_heads * C, head_size)，其中 head_size = hidden_size / num_heads。
    """

    def __init__(self, num_attention_heads, input_size, hidden_size):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def trans_to_multiple_heads(self, x):
        # x: (B, C, hidden_size) -> (B, C, num_heads, head_size) -> (B, num_heads, C, head_size)
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        # x: (B, C, W)
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        # 跨通道维做注意力打分
        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value_heads)
        # context: (B, num_heads, C, head_size)
        shape = context.size()
        # 展平 heads*C -> (B, num_heads*C, head_size)
        context = context.contiguous().view(shape[0], -1, shape[-1])
        return context


class AMC_Net(nn.Module):
    """
    AMC-Net 主网络（PyTorch 移植版）。

    forward 约定：
      输入 x: (B, 2, sig_len)，channel 0 = I，channel 1 = Q
      输出 logits: (B, num_classes)（不加 softmax）
    """

    def __init__(self,
                 num_classes=11,
                 sig_len=128,
                 extend_channel=36,
                 latent_dim=512,
                 num_heads=2,
                 conv_chan_list=None):
        super().__init__()
        self.sig_len = sig_len
        self.extend_channel = extend_channel
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.conv_chan_list = conv_chan_list

        if self.conv_chan_list is None:
            self.conv_chan_list = [36, 64, 128, 256]
        self.stem_layers_num = len(self.conv_chan_list) - 1

        self.ACM = AdaCorrModule(self.sig_len)
        self.MSM = MultiScaleModule(self.extend_channel)
        self.FFM = FeaFusionModule(self.num_heads, self.sig_len, self.sig_len)

        self.Conv_stem = nn.Sequential()
        for t in range(0, self.stem_layers_num):
            self.Conv_stem.add_module(
                f'conv_stem_{t}',
                Conv_Block(self.conv_chan_list[t], self.conv_chan_list[t + 1]),
            )

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes),
        )

    def forward(self, x):
        # x: (B, 2, sig_len)
        x = x.unsqueeze(1)                       # (B, 1, 2, W)
        x = self.ACM(x)                          # 自适应校正，(B, 1, 2, W)
        x = x / x.norm(p=2, dim=-1, keepdim=True)  # 沿 W 维 L2 归一化
        x = self.MSM(x)                          # 多尺度卷积，(B, 36, 1, W)
        x = self.Conv_stem(x)                    # 卷积主干，(B, 256, 1, W)
        x = self.FFM(x.squeeze(2))               # 特征融合，(B, 512, head_size)
        x = self.GAP(x)                          # (B, 512, 1)
        y = self.classifier(x.squeeze(2))        # (B, num_classes) logits
        return y


def build_amcnet_torch_model(input_shape=(2, 128), num_classes=11,
                             extend_channel=36, latent_dim=512, num_heads=2,
                             conv_chan_list=None):
    """
    工厂函数：构造 PyTorch 版 AMC-Net。

    Args:
        input_shape: 输入形状 (2, sig_len)，channel 0 = I，channel 1 = Q
        num_classes: 调制类别数（默认 11）
        extend_channel: 多尺度模块扩展通道数（默认 36，需能被 3 整除）
        latent_dim: 分类头隐藏维度，必须等于 num_heads * conv_chan_list[-1]（默认 512=2*256）
        num_heads: 注意力头数（默认 2）
        conv_chan_list: 卷积主干通道列表（默认 [36, 64, 128, 256]）

    Returns:
        nn.Module，forward 接收 (B, 2, sig_len)，输出 logits (B, num_classes)
    """
    sig_len = int(input_shape[1])
    return AMC_Net(
        num_classes=num_classes,
        sig_len=sig_len,
        extend_channel=extend_channel,
        latent_dim=latent_dim,
        num_heads=num_heads,
        conv_chan_list=conv_chan_list,
    )


# 别名，与项目其他 *_torch_model.py 的导出习惯保持一致
build_amcnet_torch = build_amcnet_torch_model


if __name__ == "__main__":
    model = build_amcnet_torch_model((2, 128), 11)
    x = torch.randn(4, 2, 128)
    y = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(y.shape)}")
    print(f"total params: {total:,}")
