import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .qwt import QWTForward, QWTInverse
from .QN import QuaternionNorm2d as QNorm
from .quaternion_layers import QuaternionConv as QConv
import matplotlib.pyplot as plt
import os
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, embedding_dim, **kwargs):
        super().__init__()
        self.wt = QWTForward('cuda')
        self.bn = QNorm(embedding_dim)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        yL, y_LH, y_HL, y_HH = self.wt(x)
        # F_efm = torch.sqrt(y_LH * y_LH + y_HL * y_HL + y_HH * y_HH)
        x = torch.cat([yL, y_LH, y_HL, y_HH], dim=1)
        x = self.bn(x)

        return x


class PatchMerging2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, embedding_dim, **kwargs):
        super().__init__()
        self.li = QConv(4, 2 * embedding_dim, 1, 1, 1, groups=embedding_dim//4)
        # self.li = nn.Conv2d(embedding_dim, 2 * embedding_dim, 1, groups=embedding_dim)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.li(x)
        x = self.pool(x)

        return x


class PatchExpand2D(nn.Module):

    def __init__(self, embedding_dim, **kwargs):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.li = QConv(4, embedding_dim, 1, 1, 1, groups=embedding_dim//2)
        # self.li = nn.Conv2d(2*embedding_dim, embedding_dim, 1, groups=embedding_dim)

    def forward(self, x):
        x = self.up(x)
        x = self.li(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.bn = QNorm(embedding_dim)
        self.iwt = QWTInverse('cuda')
    
    def forward(self, x):
        x = self.bn(x)
        # Separate LL, LH, HL, HH from the input tensor
        embedding_dim = x.shape[1] // 4
        yL = x[:, :embedding_dim, :, :]  # LL component
        y_HL = x[:, embedding_dim:2 * embedding_dim, :, :]  # HL component
        y_LH = x[:, 2 * embedding_dim:3 * embedding_dim, :, :]  # LH component
        y_HH = x[:, 3 * embedding_dim:, :, :]  # HH component

        # # Combine the components into a tuple
        # yH = [torch.stack((y_HL, y_LH, y_HH), dim=2)]
        #
        # # Perform the inverse wavelet transform
        # embedding_dim = x.shape[1] // 4
        # yL = x[:, :embedding_dim, :, :]  # LL component
        # y_HL = x[:, embedding_dim:2 * embedding_dim, :, :]  # HL component
        # y_LH = x[:, 2 * embedding_dim:3 * embedding_dim, :, :]  # LH component
        # y_HH = x[:, 3 * embedding_dim:, :, :]  # HH component
        # channel_order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        # yL = yL[:, channel_order, :, :]
        # y_HL = y_HL[:, channel_order, :, :]
        # y_LH = y_LH[:, channel_order, :, :]
        # y_HH = y_HH[:, channel_order, :, :]
        x = self.iwt(yL, y_HL, y_LH, y_HH)

        return x


class QDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(QDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = QConv(4, c_in, k_size, stride, padding=padding, groups=c_in // 4)
        self.pw = QConv(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class IQDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IQDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = QConv(4, c_out, k_size, stride, padding=padding, groups=c_out // 4)
        self.pw = QConv(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out


class QAttn(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, dilation=1):
        super().__init__()

        gc = int(in_channels // 4)  # channel numbers of a convolution branch
        self.dwconv_hw = QConv(4, gc, square_kernel_size, padding=square_kernel_size // 2, stride=1, groups=gc//4)
        self.dwconv_w = QConv(4, gc, kernel_size=(1, band_kernel_size),
                              padding=(0, (band_kernel_size - 1) * dilation // 2), stride=1,
                              dilatation=dilation, groups=gc//4)
        self.dwconv_h = QConv(4, gc, kernel_size=(band_kernel_size, 1),
                              padding=((band_kernel_size - 1) * dilation // 2, 0), stride=1,
                              dilatation=dilation, groups=gc//4)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
    
    def forward(self, x):
        # Reassigning the processing order as per the new requirement
        x_id, x_h, x_w, x_hw = torch.split(x, self.split_indexes, dim=1)

        return torch.cat(
            (x_id, self.dwconv_h(x_h), self.dwconv_w(x_w), self.dwconv_hw(x_hw)),
            dim=1,
        )


class QFFN(nn.Module):
    def __init__(self, dim, h_dim=None, out_dim=None):
        super().__init__()
        self.h_dim = dim * 2 if h_dim == None else h_dim
        self.out_dim = dim if out_dim == None else out_dim

        self.act = nn.GELU()
        self.fc1 = QDSC(dim, self.h_dim)
        self.norm = QNorm(self.out_dim)
        self.fc2 = QDSC(self.h_dim, self.h_dim)
        self.fc3 = IQDSC(self.h_dim, self.out_dim)
    
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.act(self.fc3(self.act(self.fc2(self.act(self.fc1(x))))))
        x = self.norm(x)
        return x


class QTBlock(nn.Module):
    def __init__(
            self,
            dim: int = 0,
            drop_path: float = 0,
            attn_drop_rate: float = 0,
            **kwargs,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path)
        self.attn_drop = DropPath(attn_drop_rate)
        self.glo = QFFN(dim)
        self.loc = QAttn(dim)
        self.ln1 = QN(dim)
        self.ln2 = QN(dim)

    def forward(self, x):
        x = self.attn_drop(self.loc(self.ln1(x))) + x
        x = self.drop_path(self.glo(self.ln2(x))) + x

        return x


class QTLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=QN,
            downsample=None,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        if depth == 2:
            self.res = True
        else:
            self.res = False
        self.blocks = nn.ModuleList([
            QTBlock(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim)
        else:
            self.downsample = None

    def forward(self, x):
        x_vice = x
        for blk in self.blocks:
            x = blk(x)
        if self.res is True:
            x = x + x_vice
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class QTLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        if depth == 2:
            self.res = True
        else:
            self.res = False

        self.blocks = nn.ModuleList([
            QTBlock(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
            )
            for i in range(depth)])

        if upsample is not None:
            self.upsample = upsample(dim)
        else:
            self.upsample = None
    
    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        x_vice = x
        for blk in self.blocks:
            x = blk(x)
        if self.res is True:
            x = x + x_vice
        return x


# def visualize_features(features, stage, save_dir="feature_maps"):
#     """可视化和保存特征图
#     Args:
#         features (torch.Tensor): 输入特征图 (B, C, H, W)
#         stage (str): 当前阶段名称（如 "layer1", "layer_up1"）
#         save_dir (str): 保存特征图的目录
#     """
#     import os
#     import matplotlib.pyplot as plt
#
#     os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
#     b, c, h, w = features.shape
#
#     # 对通道进行平均，合并为单通道特征图
#     features_mean = features[0].mean(dim=0).cpu().detach().numpy()  # (H, W)
#
#     # 归一化到 [0, 1]
#     features_mean = (features_mean - features_mean.min()) / (features_mean.max() - features_mean.min() + 1e-5)
#
#     # 保存图像
#     plt.imshow(features_mean, cmap="viridis")
#     plt.colorbar()
#     plt.title(f"{stage} - Mean Feature Map")
#     plt.axis("off")
#     plt.savefig(os.path.join(save_dir, f"{stage}.png"))
#     plt.close()
#
#     print(f"Saved: {os.path.join(save_dir, f'{stage}.png')}")

# def visualize_features(features, stage, save_dir="feature_maps"):
#     """可视化和保存三通道特征图
#     Args:
#         features (torch.Tensor): 输入特征图 (B, C, H, W)
#         stage (str): 当前阶段名称（如 "layer1", "layer_up1"）
#         save_dir (str): 保存特征图的目录
#     """
#     import os
#     import matplotlib.pyplot as plt
#     import numpy as np

#     os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
#     b, c, h, w = features.shape

#     # 假设你需要可视化的是第一个样本（b=0），且特征图是三通道的
#     features = features[0].cpu().detach().numpy()  # (C, H, W)

#     # 确保是三通道特征图
#     if features.shape[0] == 3:
#         # 归一化每个通道到 [0, 1]
#         features = np.clip(features, features.min(axis=(1, 2), keepdims=True), features.max(axis=(1, 2), keepdims=True))
#         features = (features - features.min(axis=(1, 2), keepdims=True)) / (
#                     features.max(axis=(1, 2), keepdims=True) - features.min(axis=(1, 2), keepdims=True) + 1e-5)

#         # 将三个通道合并成一个 (H, W, 3) 的图像
#         features_rgb = np.transpose(features, (1, 2, 0))  # (H, W, C)

#         # 保存图像
#         plt.imshow(features_rgb)
#         plt.axis("off")
#         plt.savefig(os.path.join(save_dir, f"{stage}.png"), transparent=True, bbox_inches='tight', pad_inches=0)
#         plt.close()

#         print(f"Saved: {os.path.join(save_dir, f'{stage}.png')}")
#     else:
#         print(f"Warning: Expected 3 channels, but got {features.shape[0]} channels.")


class QTUNet(nn.Module):
    def __init__(self, num_classes=1, depths=[1, 2, 2, 1], depths_decoder=[1, 2, 2, 1],
                 dims=[48, 96, 192, 384], dims_decoder=[384, 192, 96, 48],
                 attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=QN, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(embedding_dim=self.embed_dim)

        # WASTED absolute position embedding ======================
        # self.ape = False
        # drop_rate = 0.0

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = QTLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = QTLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(embedding_dim=self.embed_dim)
        self.final_conv = nn.Conv2d(3, num_classes, 1)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward_features(self, x):
        skip_list = []
        feature_maps = []  # 用于存储 self.layers 的特征图

        x = self.patch_embed(x)
        feature_maps.append(x)  # 存储 patch_embed 后的特征图

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
            feature_maps.append(x)  # 存储每个 layer 后的特征图

        return x, skip_list, feature_maps
    
    def forward_features_up(self, x, skip_list):
        feature_maps_up = []  # 用于存储 self.layers_up 的特征图
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])

            feature_maps_up.append(x)  # 存储每个 layer_up 后的特征图

        return x + skip_list[-4], feature_maps_up
    
    def forward_final(self, x):
        x = self.final_up(x)
        # visualize_features(x, 'iqwt')
        x = self.final_conv(x)
        return x
    
    def forward(self, x):
        x, skip_list, feature_maps = self.forward_features(x)

        # 可视化 self.layers 的特征图
        # for i, feature in enumerate(feature_maps):
        #     visualize_features(feature, f"layer{i + 1}")

        x, feature_maps_up = self.forward_features_up(x, skip_list)

        # 可视化 self.layers_up 的特征图
        # for i, feature in enumerate(feature_maps_up):
        #     visualize_features(feature, f"layer_up{i + 1}")

        x = self.forward_final(x)

        # 最终输出可以选择是否可视化
        # visualize_features(x, "final_output")

        return x

# x = torch.randn(32, 3, 256, 256).to('cuda')
# V = QTUNet()
# total_params = sum(p.numel() for p in V.parameters())
# print(f"Total parameters in RENet: {total_params}")
