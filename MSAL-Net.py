import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelTimeAttention(nn.Module):
    """双路径注意力机制：通道注意力 + 频率轴注意力"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        # 确保reduction不会使中间层通道数小于1
        mid_channels = max(1, channel // reduction)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, channel, 1),
            nn.Sigmoid()
        )
        self.time_att = nn.Sequential(
            nn.Conv1d(channel, mid_channels, 1),
            nn.GELU(),
            nn.Conv1d(mid_channels, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        ta = x.mean(dim=2)
        ta = self.time_att(ta).unsqueeze(2)
        return x * ca * ta

class MultiScaleDSConv(nn.Module):
    """多尺度深度可分离卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 确保out_channels至少为4，以便能够正确划分为4个分支
        assert out_channels >= 4, "Output channels must be at least 4 for MultiScaleDSConv"

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.BatchNorm2d(out_channels // 4),
                nn.GELU(),
                nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, groups=out_channels // 4),
                nn.Conv2d(out_channels // 4, out_channels // 4, 1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            )
        ])

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class FreqTemporalBlock(nn.Module):
    """频率-时间特征块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.multi_scale = MultiScaleDSConv(in_channels, out_channels)
        self.attention = ChannelTimeAttention(out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = self.multi_scale(x)
        x = self.attention(x)
        return F.gelu(x + residual)


class PlainConvBlock(nn.Module):
    """普通卷积块（有残差连接）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = self.bn(self.conv(x))
        return F.gelu(x + residual)





# 完整模型
class Net(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(Net, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),  # 深度卷积，保持通道数
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )

        # 修改blocks层，起始通道数为19
        self.blocks = nn.Sequential(
            FreqTemporalBlock(in_channel, 64),  # 19 -> 64
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlock(64, 128),  # 64 -> 128
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlock(128, 256),  # 128 -> 256
        )

        # 修改LSTM的输入维度
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, dropout=dropout, num_layers=2, batch_first=True,
                            bidirectional=True)

        # 修改分类器的输入维度
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),  # 2*256 = 512 -> 256
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.stem_output = None

    def forward(self, x, labels=None, save_stem_output=False):
        x = self.stem(x)

        self.stem_output = x
        if save_stem_output:
            np.save('stem_output.npy', x.detach().cpu().numpy())
            print("Stem output saved as 'stem_output.npy'.")
            if labels is not None:
                np.save('labels.npy', labels.detach().cpu().numpy())
                print("Labels saved as 'labels.npy'.")

        x = self.blocks(x).mean(2).transpose(1, 2)
        out, (h_n, c_n) = self.lstm(x)
        h_n = h_n.transpose(0, 1).reshape(x.size(0), -1)
        return self.classifier(h_n)

# 纯粹的传统卷积块（无残差连接）
class PureConvBlock(nn.Module):
    """纯粹的传统卷积块（无残差连接）"""

    def __init__(self, in_channels, out_channels):
        super(PureConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
# 消融1 - 只移除LSTM，保留所有其他组件
class NetNoLSTM(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(NetNoLSTM, self).__init__()
        # 保持与Net相同的stem层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # 保持与Net相同的blocks层
        self.blocks = nn.Sequential(
            FreqTemporalBlock(in_channel, 64),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlock(64, 128),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlock(128, 256),
        )
        # 移除LSTM，使用全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 添加这一行，初始化stem_output属性
        self.stem_output = None

    def forward(self, x):
        x = self.stem(x)

        # 添加这一行，保存stem输出
        self.stem_output = x

        x = self.blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)

# 消融2：移除LSTM和注意力机制 - 添加这个缺失的类
class FreqTemporalBlockNoAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.multi_scale = MultiScaleDSConv(in_channels, out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = self.multi_scale(x)
        return F.gelu(x + residual)

# 消融2：移除LSTM和注意力机制
class NetNoLSTMNoAttention(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(NetNoLSTMNoAttention, self).__init__()
        # 保持与Net相同的stem层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # 使用无注意力的blocks
        self.blocks = nn.Sequential(
            FreqTemporalBlockNoAttention(in_channel, 64),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlockNoAttention(64, 128),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlockNoAttention(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 添加stem_output属性
        self.stem_output = None

    def forward(self, x):
        x = self.stem(x)

        # 保存stem输出
        self.stem_output = x

        x = self.blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


# 消融3：移除LSTM、注意力机制和多尺度卷积 (保留残差连接)
class NetNoLSTMNoAttentionNoMSConv(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(NetNoLSTMNoAttentionNoMSConv, self).__init__()
        # 保持与Net相同的stem层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # 使用普通卷积块（有残差连接）
        self.blocks = nn.Sequential(
            PlainConvBlock(in_channel, 64),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            PlainConvBlock(64, 128),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            PlainConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 添加stem_output属性
        self.stem_output = None

    def forward(self, x):
        x = self.stem(x)

        # 保存stem输出
        self.stem_output = x

        x = self.blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


# 消融4：基础CNN (无LSTM、无注意力、无多尺度卷积、无残差连接)
class NetPureCNN(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(NetPureCNN, self).__init__()
        # 保持与Net相同的stem层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # 使用无残差连接的纯卷积块
        self.blocks = nn.Sequential(
            PureConvBlock(in_channel, 64),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            PureConvBlock(64, 128),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            PureConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 添加stem_output属性
        self.stem_output = None

    def forward(self, x):
        x = self.stem(x)
        # 保存stem输出
        self.stem_output = x
        x = self.blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


if __name__ == '__main__':
    import torchinfo


