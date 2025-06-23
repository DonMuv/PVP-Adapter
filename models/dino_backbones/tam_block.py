import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TAMBlock2D(nn.Module):
    def __init__(self, in_channels=1024, ratio=2):
        super(TAMBlock2D, self).__init__()
        self.ratio = ratio
        self.shared_layer_one = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.shared_layer_two = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, stride=1, padding=0)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 2D Avg Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 2D Max Pooling

    def forward(self, input_feature):
        # new 
        input_feature = input_feature.permute(1, 0, 2)
        cls_token, input_feature = torch.tensor_split(input_feature, [1], dim=0)
        N, B, C = input_feature.shape
        h = int(np.sqrt(N))
        w = h
        input_feature = input_feature.reshape(B, C, h, w)
        
        channel = input_feature.size(1)  # Assuming the input is of shape (batch, channels, height, width)

        # Avg Pool path
        avg_pool = self.avg_pool(input_feature)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        # Max Pool path
        max_pool = self.max_pool(input_feature)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        # Combine both paths
        cbam_feature = avg_pool + max_pool
        cbam_feature = torch.sigmoid(cbam_feature)

        # Reshape to match input size
        cbam_feature = cbam_feature.view(-1, channel, 1, 1)

        # Apply feature to input
        res = input_feature * cbam_feature
        res = res.reshape(N, B, C)
        res = torch.cat([cls_token, res], dim=0)
        res = res.permute(1, 0, 2)
        return res

# Example usage:
# Assuming input_feature is a 4D tensor of shape (batch_size, channels, height, width)


# input_feature = torch.randn(8, 64, 32, 32)  # Example input
# tam_block_2d = TAMBlock2D(in_channels=64)
# output = tam_block_2d(input_feature)
