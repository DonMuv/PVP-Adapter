
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

class HfcEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 1,
        embed_dim: int = 1024,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        # x = x.permute(0, 2, 3, 1)
        # x = x.flatten(2).transpose(1, 2)  # B HW C
        # x = self.norm(x)
        return x


class CrossAttentionHfcPatch(nn.Module):
    """
    attend patch embeddings with high frequency componenets
    """

    def __init__(
            self,
            d_model=1024,
            nhead=8,
            dropout=0.1,
            dim_feedforward=1024,
            activation='relu',
            proj_dim=1024
    ):
        super().__init__()
        self.activation = F.relu
        self.proj_hfc = nn.Conv2d(d_model, proj_dim, (1, 1))
        self.proj_patch = nn.Conv2d(d_model, proj_dim, (1, 1))
        self.cross_attn = nn.MultiheadAttention(proj_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(proj_dim, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(dim_feedforward)

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, proj_dim, 32, 32))

    def forward(
            self,
            hfc_embed,
            patch_embed,
    ):
        b, c, h, w = hfc_embed.shape
        hfc_embed = self.proj_hfc(hfc_embed) + self.pos_embed
        patch_embed = self.proj_patch(patch_embed)
        # flatten NxCxHxW to HWxNxC
        hfc_embed = hfc_embed.flatten(2).permute(2, 0, 1)
        patch_embed = patch_embed.flatten(2).permute(2, 0, 1)

        # # fix
        # b, hw, c = hfc_embed.shape
        # hfc_embed = self.proj_hfc(hfc_embed) + self.pos_embed
        # patch_embed = self.proj_patch(patch_embed)


        src2 = self.cross_attn(
            query=patch_embed,
            key=hfc_embed,
            value=hfc_embed)[0]
        patch_embed = patch_embed + self.dropout1(src2)
        patch_embed = self.norm1(patch_embed)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(patch_embed))))
        # src2 = self.activation(self.linear1(patch_embed))
        src2 = src2 + self.dropout3(patch_embed)
        patch_embed = self.norm2(src2)

        patch_embed = patch_embed.permute(1, 0, 2).view(b, c, h, w)

        return patch_embed


