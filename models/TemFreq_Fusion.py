import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Gabor Wavelet Activation Function
class GaborActivation(nn.Module):
    def __init__(self, in_channels):
        super(GaborActivation, self).__init__()
        self.in_channels = in_channels

        # 频率参数 omega0（可以学习）
        self.omega0 = nn.Parameter(torch.randn(in_channels, 1, 1))  # 频率参数
        self.sigma = nn.Parameter(torch.randn(in_channels, 1, 1))   # 标准差

    def forward(self, x):
        # 计算 Gabor 激活
        gabor_filter = torch.exp(-self.sigma * x**2) * torch.cos(self.omega0 * x)
        return x * gabor_filter  # 进行非线性变换

# Temporal Frequency Fusion (TFF)
class TFF(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024, hidden_dim=1024):
        super(TFF, self).__init__()
        
        # Pre 输入通道 + Post 输入通道
        self.pre_conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.post_conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)  

        # Gabor 激活函数
        self.gabor_activation = GaborActivation(hidden_dim)

        # 交互融合模块
        self.spa_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.fre_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # 最终的融合层
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, pre_feature, post_feature):
        """
        输入：
        - spa_feature: 空间特征，尺寸 [B, C, H, W]
        - fre_feature: 频域特征，尺寸 [B, C, H, W]
        """

        # 
        pre_out = self.pre_conv1(pre_feature)
        pre_out = self.gabor_activation(pre_out)  # 频域通过 Gabor 小波激活

        # 
        post_out = self.post_conv1(post_feature)
        post_out = self.gabor_activation(post_out)  # 频域通过 Gabor 小波激活

        # 交互融合
        pre_out = self.pre_conv2(pre_out)
        post_out = self.post_conv2(post_out)
        # combined = pre_out + post_out  # 融合空间特征和频域特征
        combined = torch.matmul(pre_out, post_out)

        # 最终解码
        output = self.final_conv(combined)
        return output
