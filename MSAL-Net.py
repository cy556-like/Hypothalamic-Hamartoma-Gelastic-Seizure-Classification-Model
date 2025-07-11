import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelTimeAttention(nn.Module):
    """Dual-path attention mechanism: Channel attention + Frequency-axis attention"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        # Ensure reduction does not make the intermediate channel count less than 1
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
    """Multi-scale depthwise separable convolution block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Ensure out_channels is at least 4 to properly split into 4 branches
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
    """Frequency-temporal feature block"""

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
    """Plain convolution block (with residual connection)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = self.bn(self.conv(x))
        return F.gelu(x + residual)


# Complete model
class Net(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(Net, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),  # Depthwise convolution, keeping channel count
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )

        # Modify blocks layer, starting with 19 channels
        self.blocks = nn.Sequential(
            FreqTemporalBlock(in_channel, 64),  # 19 -> 64
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlock(64, 128),  # 64 -> 128
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlock(128, 256),  # 128 -> 256
        )

        # Modify LSTM input dimension
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, dropout=dropout, num_layers=2, batch_first=True,
                            bidirectional=True)

        # Modify classifier input dimension
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


# Pure traditional convolution block (without residual connection)
class PureConvBlock(nn.Module):
    """Pure traditional convolution block (without residual connection)"""

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


# Ablation 1 - Remove only LSTM, keep all other components
class NetNoLSTM(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(NetNoLSTM, self).__init__()
        # Keep the same stem layer as Net
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # Keep the same blocks layer as Net
        self.blocks = nn.Sequential(
            FreqTemporalBlock(in_channel, 64),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlock(64, 128),
            nn.MaxPool2d((3, 1), (2, 1), (1, 0)),
            FreqTemporalBlock(128, 256),
        )
        # Remove LSTM, use global pooling instead
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Add this line to initialize stem_output attribute
        self.stem_output = None

    def forward(self, x):
        x = self.stem(x)

        # Add this line to save stem output
        self.stem_output = x

        x = self.blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


# Ablation 2: Remove LSTM and attention mechanism - Add this missing class
class FreqTemporalBlockNoAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.multi_scale = MultiScaleDSConv(in_channels, out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = self.multi_scale(x)
        return F.gelu(x + residual)


# Ablation 2: Remove LSTM and attention mechanism
class NetNoLSTMNoAttention(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(NetNoLSTMNoAttention, self).__init__()
        # Keep the same stem layer as Net
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # Use blocks without attention
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

        # Add stem_output attribute
        self.stem_output = None

    def forward(self, x):
        x = self.stem(x)

        # Save stem output
        self.stem_output = x

        x = self.blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


# Ablation 3: Remove LSTM, attention mechanism, and multi-scale convolution (retain residual connections)
class NetNoLSTMNoAttentionNoMSConv(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(NetNoLSTMNoAttentionNoMSConv, self).__init__()
        # Keep the same stem layer as Net
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # Use plain convolution blocks (with residual connections)
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

        # Add stem_output attribute
        self.stem_output = None

    def forward(self, x):
        x = self.stem(x)

        # Save stem output
        self.stem_output = x

        x = self.blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


# Ablation 4: Basic CNN (no LSTM, no attention, no multi-scale convolution, no residual connections)
class NetPureCNN(nn.Module):
    def __init__(self, in_channel, dropout=0.5, num_classes=2):
        super(NetPureCNN, self).__init__()
        # Keep the same stem layer as Net
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(7, 3), padding=(3, 1), groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # Use pure convolution blocks without residual connections
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

        # Add stem_output attribute
        self.stem_output = None

    def forward(self, x):
        x = self.stem(x)
        # Save stem output
        self.stem_output = x
        x = self.blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


if __name__ == '__main__':
    import torchinfo