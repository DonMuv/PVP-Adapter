import lightning as L
import torch
import torch.nn as nn

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_


class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x
    

class TSLA_PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = 16
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if True:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true

        x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # # If only ICB is true
        # elif args.ICB:
        #     x = x + self.drop_path(self.icb(self.norm2(x)))
        # # If only ASB is true
        # elif args.ASB:
        #     x = x + self.drop_path(self.asb(self.norm1(x)))
        # # If neither is true, just pass x through
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

