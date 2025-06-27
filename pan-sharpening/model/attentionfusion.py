from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils_att import ChannelPool
from timm.models.layers import DropPath, Mlp, trunc_normal_
from timm.models.helpers import named_apply
from functools import partial
from pos_utils import get_2d_sincos_pos_embed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):

        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h * t_w, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class AttentionFusionBlock(nn.Module):
    def __init__(self, img_size_uav, img_size_satellite, patch_size=16, dropout_rate=0.1, input_ndim=768,
                 mid_ndim=[768, 512, 256], attention_layer_num=12):
        super().__init__()
        self.img_size_uav=img_size_uav
        self.img_size_satellite=img_size_satellite
        self.patch_size = patch_size
        self.dropout_rate = dropout_rate
        self.input_ndim = input_ndim
        self.mid_ndim = mid_ndim
        self.attention_layer_num = attention_layer_num

        self.grid_size_uav = self.img_size_uav//self.patch_size
        self.grid_size_satellite = self.img_size_satellite//self.patch_size
        self.num_patches_uav = self.grid_size_uav ** 2
        self.num_patches_satellite = self.grid_size_satellite ** 2

        self.pos_drop = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, attention_layer_num)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=input_ndim, num_heads=4, mlp_ratio=4, qkv_bias=True,
                drop=dropout_rate, attn_drop=dropout_rate, drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(attention_layer_num)])
        self.out_linears = nn.ModuleList()
        last_ndim = input_ndim
        for channel in mid_ndim:
            self.out_linears.append(nn.Linear(last_ndim, channel))
            last_ndim = channel

        self.pos_embed_uav = nn.Parameter(
            torch.zeros(1, self.num_patches_uav, input_ndim),
            requires_grad=False)

        self.pos_embed_satellite = nn.Parameter(
            torch.zeros(1, self.num_patches_satellite, input_ndim),
            requires_grad=False)
        self.init_pos_embed()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_weights_vit_timm(self, module: nn.Module, name: str):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, 'init_weights'):
            module.init_weights()

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed_uav = get_2d_sincos_pos_embed(
            self.pos_embed_uav.shape[-1],
            int(self.num_patches_uav ** .5),
            cls_token=False)

        self.pos_embed_uav.data.copy_(torch.from_numpy(pos_embed_uav).float().unsqueeze(0))

        pos_embed_satellite = get_2d_sincos_pos_embed(
            self.pos_embed_satellite.shape[-1],
            int(self.num_patches_satellite ** .5),
            cls_token=False)
        self.pos_embed_satellite.data.copy_(
            torch.from_numpy(pos_embed_satellite).float().unsqueeze(0))

    def forward_single(self, z, x):
        # B, C, H_uav, W_uav = z.shape if len(z.shape) == 4 else (z.shape[0], z.shape[1], 1, z.shape[2])
        # B, C, _, _ = z.shape
        B = z.shape[0]
        C = 4
        H_uav = W_uav = self.grid_size_uav
        H_satellite = W_satellite = self.grid_size_satellite
        # H_uav = W_uav =256
        # H_satellite = W_satellite =256

        # convert B,C,H,W->B,N,C
        #print("Z0",z.shape)
        z = z.flatten(2).transpose(1, 2).contiguous()
        x = x.flatten(2).transpose(1, 2).contiguous()
        # position embedding


        # z=z.unsqueeze(0)

        # z=self.input_conv(z)
        # z = z.view(1, 256, -1)

        # z=self.conv1d_layer(z)
        #print("z",z.shape)
        #print("self.pos_embed_uav", self.pos_embed_uav.shape)
        x_uav = z + self.pos_embed_uav

        # x = self.input_conv(x)
        # x = x.view(1, 256, -1)

        # x = self.conv1d_layer(x)
        #print("x",x.shape)
        #print("self.pos_embed_satellite",self.pos_embed_satellite.shape)
        x_satellite = x + self.pos_embed_satellite
        x = torch.cat([x_uav, x_satellite], dim=1)
        #print("x5", x.shape)
        x = self.pos_drop(x)
        z = self.pos_drop(z)

        #print("x6", x.shape)
        # attention block
        for blk in self.blocks:
            x = blk(x, H_uav, W_uav, H_satellite, W_satellite)

        x_uav, x_satellite = torch.split(x, [H_uav * W_uav, H_satellite * W_satellite], dim=1)

        #print("x_satelliteaa", x_satellite.shape)
        #print("self.out_linears", self.out_linears)
        #print("len(self.out_linears)", len(self.out_linears))
        for ind in range(len(self.out_linears)):
            x_satellite = self.out_linears[ind](x_satellite)

        # print("x_satellite0", x_satellite.shape)
        # x_uav_2d = x_uav.transpose(1, 2).reshape(B, C, H_uav, W_uav)
        # print("x_uav_2d",x_uav_2d.shape)
        #print("x_satellite", x_satellite.shape)
        #print("H_satellite",H_satellite)
        #print("x_uav",x_uav.shape)
        #print("H_uav",H_uav)
        x_uav_2d = x_uav.transpose(1, 2).reshape(B, 256, H_uav, W_uav)
        x_satellite_2d = x_satellite.transpose(1, 2).reshape(B, 256,  H_satellite,  H_satellite)

        #return x_satellite_2d, None

        return x_satellite_2d, x_uav_2d

    def forward(self, z, x):
        return self.forward_single(z, x)


# 假设 Block 类和 get_2d_sincos_pos_embed 函数已经在其他地方定义
# from your_module import Block, get_2d_sincos_pos_embed

# 1. 定义模型配置
# print(opt.data_config["UAVhw"][0])
# 2. 实例化 AttentionFusionBlock
if __name__ == "__main__":
    model = AttentionFusionBlock(
        img_size_uav=128,
        img_size_satellite=64,
        patch_size=16,
        dropout_rate=0.1,
        input_ndim=256,
        mid_ndim=[512, 512],
        attention_layer_num=12
    )

    # 创建示例输入


    # 3. 创建示例输入
    batch_size = 2
    uav_features = torch.randn(batch_size, 512, 32, 32)  # 无人机图像特征
    satellite_features = torch.randn(batch_size, 512, 32, 32)  # 卫星图像特征

    # 4. 前向传播
    satellite_output, _ = model(uav_features, satellite_features)

    # 5. 输出结果
    print("卫星图像输出特征形状:", satellite_output.shape)