"""PyTorch MCLDNN (Multi-Channel LDNN) for benchmark.

忠实移植 Keras 版 mcldnn_model.py 的 MCLDNN 架构。

原 Keras 版为多输入(完整 IQ 的 2D 卷积分支 + I/Q 各自的 1D 卷积分支),
本 PyTorch 版统一接收单个张量 x，形状 (B, 2, 128):
    channel 0 = I, channel 1 = Q
forward 内部派生出各分支所需输入，输出 logits (B, num_classes)，不含 softmax。

数据流对照(以 seq_len=128 为例)，PyTorch 内部一律用 channels-first：
  Path1  : 完整 IQ -> (B,1,2,L)   -> Conv2d(1->50, k=(2,8), same, relu)   -> (B,50,2,L)
  Path2  : I       -> (B,1,L)     -> Conv1d(1->50, k=8, causal, relu)      -> (B,50,L) -> (B,50,1,L)
  Path3  : Q       -> (B,1,L)     -> Conv1d(1->50, k=8, causal, relu)      -> (B,50,L) -> (B,50,1,L)
  concat(P2,P3) 沿"高"维 -> (B,50,2,L) -> Conv2d(50->50, k=(1,8), same, relu) -> (B,50,2,L)
  concat(P1,x_iq) 沿通道维 -> (B,100,2,L)
  Conv2d(100->100, k=(2,5), valid, relu) -> (B,100,1,L-4)
  -> reshape 成 (B, L-4, 100) 喂给 LSTM
  LSTM(100->128, seq) -> LSTM(128->128) -> Dense(128,selu) -> Dropout(0.5)
  -> Dense(128,selu) -> Dropout(0.5) -> Dense(num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv2d_same_pad(kernel_size):
    """对应 Keras Conv2D padding='same'(stride=1)的对称 padding。

    返回 (pad_h, pad_w)，沿用 (kernel-1)//2 的左上 padding。配合 stride=1
    且核尺寸为偶数时(如 k=8)，需要在右/下额外补 1，因此这里返回的是
    F.pad 用的非对称四元组 (left, right, top, bottom)。
    """
    kh, kw = kernel_size
    pad_top = (kh - 1) // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)


class MCLDNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # input_shape = (2, seq_len)
        self.seq_len = input_shape[1]

        # --- Path 1: 完整 IQ 的 2D 卷积分支 ---
        # Keras: Conv2D(50, (2,8), padding='same', relu)，输入通道 1
        self._pad_p1 = _conv2d_same_pad((2, 8))
        self.conv2d_1 = nn.Conv2d(1, 50, kernel_size=(2, 8))

        # --- Path 2 / Path 3: I、Q 各自的 1D 卷积分支 ---
        # Keras: Conv1D(50, 8, padding='causal', relu)，输入通道 1
        # causal: 左侧补 (kernel-1) 个 0，输出长度不变
        self._causal_pad = 8 - 1
        self.conv1d_i = nn.Conv1d(1, 50, kernel_size=8)
        self.conv1d_q = nn.Conv1d(1, 50, kernel_size=8)

        # 拼接后的 I/Q 特征再过一层 2D 卷积
        # Keras: Conv2D(50, (1,8), padding='same', relu)，输入通道 50
        self._pad_iq = _conv2d_same_pad((1, 8))
        self.conv2d_iq = nn.Conv2d(50, 50, kernel_size=(1, 8))

        # --- 所有分支拼接后的最终空间卷积 ---
        # Keras: Conv2D(100, (2,5), padding='valid', relu)，输入通道 100
        self.conv2d_final = nn.Conv2d(100, 100, kernel_size=(2, 5))

        # 经 valid 卷积后高维 2->1，宽维 L->L-4
        self.temporal_dim = self.seq_len - 5 + 1

        # --- 时序部分 ---
        # Keras LSTM 默认接收 (batch, time, feature)，batch_first=True
        self.lstm_1 = nn.LSTM(input_size=100, hidden_size=128, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # --- 分类头 ---
        self.dense_1 = nn.Linear(128, 128)
        self.dropout_1 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(128, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 2, L)，channel 0 = I, channel 1 = Q
        i = x[:, 0:1, :]  # (B, 1, L)
        q = x[:, 1:2, :]  # (B, 1, L)

        # --- Path 1: 完整 IQ 2D 卷积 ---
        # (B, 2, L) -> (B, 1, 2, L)：高=2(I/Q), 宽=L
        x1 = x.unsqueeze(1)
        x1 = F.pad(x1, self._pad_p1)
        x1 = F.relu(self.conv2d_1(x1))  # (B, 50, 2, L)

        # --- Path 2: I 通道 1D causal 卷积 ---
        x2 = F.pad(i, (self._causal_pad, 0))  # 左侧补 0
        x2 = F.relu(self.conv1d_i(x2))        # (B, 50, L)
        x2 = x2.unsqueeze(2)                  # (B, 50, 1, L)

        # --- Path 3: Q 通道 1D causal 卷积 ---
        x3 = F.pad(q, (self._causal_pad, 0))
        x3 = F.relu(self.conv1d_q(x3))        # (B, 50, L)
        x3 = x3.unsqueeze(2)                  # (B, 50, 1, L)

        # 沿"高"维拼接 I/Q 处理结果 -> (B, 50, 2, L)
        x_iq = torch.cat([x2, x3], dim=2)
        x_iq = F.pad(x_iq, self._pad_iq)
        x_iq = F.relu(self.conv2d_iq(x_iq))   # (B, 50, 2, L)

        # 沿通道维拼接所有分支 -> (B, 100, 2, L)
        x = torch.cat([x1, x_iq], dim=1)

        # 最终 valid 空间卷积 -> (B, 100, 1, L-4)
        x = F.relu(self.conv2d_final(x))

        # 去掉高维并转成 (B, time=L-4, feature=100) 喂给 LSTM
        x = x.squeeze(2)              # (B, 100, L-4)
        x = x.transpose(1, 2)         # (B, L-4, 100)

        x, _ = self.lstm_1(x)         # (B, L-4, 128)
        x, _ = self.lstm_2(x)         # (B, L-4, 128)
        x = x[:, -1, :]               # 取最后一个时间步，等价 return_sequences=False

        # 分类头，Keras 用 selu
        x = F.selu(self.dense_1(x))
        x = self.dropout_1(x)
        x = F.selu(self.dense_2(x))
        x = self.dropout_2(x)
        return self.out(x)            # logits (B, num_classes)，不加 softmax


def build_mcldnn_torch_model(input_shape, num_classes):
    return MCLDNN(input_shape=input_shape, num_classes=num_classes)
