
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
from VMamba.classification.models.vmamba import VSSBlock 


from functools import partial
from typing import Optional, Callable
from pytorch_wavelets import DWTForward, DWTInverse 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from typing import Optional, Callable, Any
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
#from .refine import Refine
try:
    import selective_scan_cuda_oflex
except Exception as e:
    raise
device_id0 = 'cuda:0'


# 假设这些模块已经定义
# from some_module import VSSBlock, CMF_Mamba

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x


class Net(nn.Module):
    def __init__(self, 
                 num_channels=None, 
                 base_filter=None, 
                 args=None):
        super(Net, self).__init__()

        # 参数定义和初始化
        self.ms_channels = 4
        self.pan_channels = 1
        self.base_filter = base_filter if base_filter is not None else 32
        self.stride = 1
        self.patch_size = 1
        self.embed_dim = self.base_filter * self.stride * self.patch_size

        self.drop_path_rate = 0.0
        self.mlp_ratio = 2.0
        self.base_d_state = 4

        # 序列化 -> B,N,C
        self.ms_to_token = PatchEmbed(in_chans=self.ms_channels, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=self.pan_channels, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)

        # MS和PAN的特征提取（VSSBlock） - 每个阶段有 4 个 VSSBlock
        self._create_feature_extraction_layers()

        # CMF_Mamba模块
        self._create_cmf_mamba_layers()

        # 输出融合层
        self.fusion_out = nn.Conv2d(self.embed_dim * 4, self.ms_channels, kernel_size=1)

    def _create_feature_extraction_layers(self):
        """创建 MS 和 PAN 的特征提取层"""
        num_blocks = 4
        self.ms_feature_extraction_level = nn.ModuleList([
            VSSBlock(hidden_dim=int(self.embed_dim), 
                     drop_path=self.drop_path_rate, 
                     norm_layer=nn.LayerNorm, 
                     attn_drop_rate=0, 
                     expand=self.mlp_ratio, 
                     d_state=int(self.base_d_state)) 
            for _ in range(num_blocks)
        ])
        self.pan_feature_extraction_level = nn.ModuleList([
            VSSBlock(hidden_dim=int(self.embed_dim), 
                     drop_path=self.drop_path_rate, 
                     norm_layer=nn.LayerNorm, 
                     attn_drop_rate=0, 
                     expand=self.mlp_ratio, 
                     d_state=int(self.base_d_state)) 
            for _ in range(num_blocks)
        ])

    def _create_cmf_mamba_layers(self):
        """创建 CMF_Mamba 层"""
        num_levels = 4
        for level in range(num_levels):
            setattr(self, f'CMF_Mamaba_level{level}', CMF_Mamba(self.embed_dim))

    def forward(self, ms, _, pan):
        # 对MS进行插值放大
        ms_bic = F.interpolate(ms, scale_factor=4)
        b, c, h, w = ms_bic.shape
        ms_f = self.ms_to_token(ms_bic)
        pan_f = self.pan_to_token(pan)

        # MS特征提取
        ms_f = self._extract_features(ms_f, self.ms_feature_extraction_level, [h, w])

        # PAN特征提取
        pan_f = self._extract_features(pan_f, self.pan_feature_extraction_level, [h, w])

        # 初始化 cmf_f_out 为零
        cmf_f_out = torch.zeros_like(ms_f)

        # # CMF_Mamba融合
        # num_levels = 4
        # for level in range(num_levels):
        #     cmf_f_level = getattr(self, f'CMF_Mamaba_level{level}')(ms_f, pan_f)
        #     cmf_f_out += cmf_f_level

        #     if level < num_levels - 1:
        #         ms_f = self.ms_feature_extraction_level[level + 1](ms_f, [h, w])
        #         pan_f = self.pan_feature_extraction_level[level + 1](pan_f, [h, w])

        # CMF_Mamba融合
        num_levels = 4
        cmf_f_list = []  # 用于存储每一级的特征
        for level in range(num_levels):
            cmf_f_level = getattr(self, f'CMF_Mamaba_level{level}')(ms_f, pan_f)
            cmf_f_list.append(cmf_f_level)

            if level < num_levels - 1:
                ms_f = self.ms_feature_extraction_level[level + 1](ms_f, [h, w])
                pan_f = self.pan_feature_extraction_level[level + 1](pan_f, [h, w])
         # 级联所有级别的特征
        # 打印列表中每个张量的形状，帮助确定级联维度
        # for i, tensor in enumerate(cmf_f_list):
            # print(f"cmf_f_list[{i}] shape: {tensor.shape}")
        cmf_f_out = torch.cat(cmf_f_list, dim=-1)  # 假设在最后一个维度上进行级联

        # 合并各层的特征
        cmf_f_out = cmf_f_out.reshape(b, cmf_f_out.shape[2], h, w)
        cmf_f_out = self.fusion_out(cmf_f_out)

        # 输出的高分辨率MS
        hrms = cmf_f_out + ms_bic

        return hrms

    def _extract_features(self, x, feature_extraction_layers, shape):
        """提取特征"""
        for layer in feature_extraction_layers:
            x = layer(x, shape)
        return x

'''
class Net(nn.Module):
    def __init__(self, 
                 num_channels=None, 
                 base_filter=None, 
                 args=None):
        super(Net, self).__init__()

        # 参数定义和初始化
        self.ms_channels = 4
        self.pan_channels = 1
        self.base_filter = base_filter if base_filter is not None else 32
        self.stride = 1
        self.patch_size = 1
        self.embed_dim = self.base_filter * self.stride * self.patch_size

        self.drop_path_rate = 0.0
        self.mlp_ratio = 2.0
        self.base_d_state = 4

        # 序列化 -> B,N,C
        self.ms_to_token = PatchEmbed(in_chans=self.ms_channels, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=self.pan_channels, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)

        # MS和PAN的特征提取（VSSBlock） - 每个阶段有 4 个 VSSBlock
        num_vss_blocks_per_stage = 4
        self._create_feature_extraction_layers(num_vss_blocks_per_stage)

        # CMF_Mamba模块
        self._create_cmf_mamba_layers()

        # 输出融合层
        self.fusion_out = nn.Conv2d(self.embed_dim, self.ms_channels, kernel_size=1)

    def _create_feature_extraction_layers(self, num_vss_blocks_per_stage):
        self.ms_feature_extraction_level = nn.ModuleList([
            VSSBlock(hidden_dim=int(self.embed_dim), 
                     drop_path=self.drop_path_rate, 
                     norm_layer=nn.LayerNorm, 
                     ssm_drop_rate=0,  # 原attn_drop_rate改为ssm_drop_rate
                     mlp_ratio=self.mlp_ratio, 
                     ssm_d_state=int(self.base_d_state)) 
            for _ in range(num_vss_blocks_per_stage)
        ])
        self.pan_feature_extraction_level = nn.ModuleList([
            VSSBlock(hidden_dim=int(self.embed_dim), 
                     drop_path=self.drop_path_rate, 
                     norm_layer=nn.LayerNorm, 
                     ssm_drop_rate=0,  # 原attn_drop_rate改为ssm_drop_rate
                     mlp_ratio=self.mlp_ratio, 
                     ssm_d_state=int(self.base_d_state)) 
            for _ in range(num_vss_blocks_per_stage)
        ])

    def _create_cmf_mamba_layers(self):
        num_cmf_mamba_layers = 4
        self.cmf_mamba_layers = nn.ModuleList([
            CMF_Mamba(self.embed_dim) for _ in range(num_cmf_mamba_layers)
        ])

    def forward(self, ms, _, pan):
        # 对MS进行插值放大
        ms_bic = F.interpolate(ms, scale_factor=4)
        b, c, h, w = ms_bic.shape
        ms_f = self.ms_to_token(ms_bic)
        pan_f = self.pan_to_token(pan)

        # MS特征提取
        ms_f = self._extract_features(ms_f, self.ms_feature_extraction_level, [h, w])

        # PAN特征提取
        pan_f = self._extract_features(pan_f, self.pan_feature_extraction_level, [h, w])

        # CMF_Mamba融合
        cmf_f_out = self._cmf_mamba_fusion(ms_f, pan_f)

        # 合并各层的特征
        cmf_f_out = cmf_f_out.reshape(b, cmf_f_out.shape[2], h, w)
        cmf_f_out = self.fusion_out(cmf_f_out)

        # 输出的高分辨率MS
        hrms = cmf_f_out + ms_bic

        return hrms

    def _extract_features(self, x, feature_extraction_layer, shape):
        for layer in feature_extraction_layer:
            x = layer(x)  # 假设VSSBlock的forward方法只需要一个输入
        return x

    def _cmf_mamba_fusion(self, ms_f, pan_f):
        cmf_f_out = torch.zeros_like(ms_f)
        for level in range(len(self.cmf_mamba_layers)):
            cmf_f_level = self.cmf_mamba_layers[level](ms_f, pan_f)
            cmf_f_out += cmf_f_level
            if level < len(self.cmf_mamba_layers) - 1:
                ms_f = self.ms_feature_extraction_level[level](ms_f)
                pan_f = self.pan_feature_extraction_level[level](pan_f)
        return cmf_f_out
'''
########################## start #################################

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x
    
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)
    
class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)
    
def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    out_norm_shape="v0",
    channel_first=False,
    # ==============================
    to_dtype=True, # True: final out to dtype
    force_fp32=False, # True: input fp32
    # ==============================
    nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
    # ==============================
    SelectiveScan=None,
    CrossScan=CrossScan,
    CrossMerge=CrossMerge,
    no_einsum=False, # replace einsum with linear or conv1d to raise throughput
    dt_low_rank=True,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
        
    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    if (not dt_low_rank):
        x_dbl = F.conv1d(x.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, -1, L), [D, 4 * N, 4 * N], dim=1)
        xs = CrossScan.apply(x)
        dts = CrossScan.apply(dts)
    elif no_einsum:
        xs = CrossScan.apply(x)
        x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
    else:
        xs = CrossScan.apply(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    
    y: torch.Tensor = CrossMerge.apply(ys)

    if channel_first:
        y = y.view(B, -1, H, W)
        if out_norm_shape in ["v1"]:
            y = out_norm(y)
        else:
            y = out_norm(y.permute(0, 2, 3, 1))
            y = y.permute(0, 3, 1, 2)
        return (y.to(x.dtype) if to_dtype else y)

    if out_norm_shape in ["v1"]: # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
    else: # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


        

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))

        self.ln_11 = norm_layer(hidden_dim)
        self.self_attention1 = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path1 = DropPath(drop_path)
        self.skip_scale1= nn.Parameter(torch.ones(hidden_dim))

        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


        # self.fpre = nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)


        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True))
        

        # self.fpre1 = nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        # self.block1 = nn.Sequential(
        #     nn.Conv2d(hidden_dim,hidden_dim,1,1,0),
        #     nn.LeakyReLU(0.1,inplace=True),
        #     nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
        #     nn.LeakyReLU(0.1, inplace=True))
        # self.linear1 = nn.Linear(hidden_dim,hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear_out = nn.Linear(hidden_dim * 3,hidden_dim)

    def forward(self, input, x_size):# input is [B,HW,C]
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        # time0 = time.time()

        prepare = rearrange(input, "b h w c -> b c h w").contiguous()
        xfm = DWTForward(J=2, mode='zero', wave='haar').to(device_id0)
        ifm = DWTInverse(mode='zero', wave='haar').to(device_id0)

        # # time1 = time.time()
        # # print(time1 - time0,'prepare')
        Yl, Yh = xfm(prepare)
        # # ttime = time.time()
        # # print(ttime - time0,'wave done')
        h00 = torch.zeros(prepare.shape).float().to(device_id0)
        for i in range(len(Yh)):
          if i == len(Yh) - 1:
            h00[:, :, :Yl.size(2), :Yl.size(3)] = Yl
            h00[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 0, :, :]
            h00[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] = Yh[i][:, :, 1, :, :]
            h00[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 2, :, :]
          else:
            h00[:, :, :Yh[i].size(3), Yh[i].size(4):] = Yh[i][:, :, 0, :, :h00.shape[3] - Yh[i].size(4)]
            h00[:, :, Yh[i].size(3):, :Yh[i].size(4)] = Yh[i][:, :, 1, :h00.shape[2] - Yh[i].size(3), :]
            h00[:, :, Yh[i].size(3):, Yh[i].size(4):] = Yh[i][:, :, 2, :h00.shape[2] - Yh[i].size(3), :h00.shape[3] - Yh[i].size(4)]
        # # ttime1 = time.time()
        # # print(ttime1 - ttime,'swap done')
        # # print(h00.shape,'ttt')
        
        h00 = rearrange(h00, "b c h w -> b h w c").contiguous()

        # # print(h00)
        # # time2 = time.time()
        # # print(time2 - time1,'wavelet')
        h11 = self.ln_11(h00)
        # # print(h11.shape,'h11shape')
        h11 = h00*self.skip_scale1 + self.drop_path1(self.self_attention1(h11))

        # # time3 = time.time()
        # # print(time3 - time2,'wavelet scan')
        h11 = rearrange(h11, "b h w c -> b c h w").contiguous()

        for i in range(len(Yh)):
          if i == len(Yh) - 1:
            Yl = h11[:, :, :Yl.size(2), :Yl.size(3)] 
            Yh[i][:, :, 0, :, :] = h11[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] 
            Yh[i][:, :, 1, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] 
            Yh[i][:, :, 2, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] 
          else:
            Yh[i][:, :, 0, :, :h11.shape[3] - Yh[i].size(4)] = h11[:, :, :Yh[i].size(3), Yh[i].size(4):] 
            Yh[i][:, :, 1, :h11.shape[2] - Yh[i].size(3), :] = h11[:, :, Yh[i].size(3):, :Yh[i].size(4)] 
            Yh[i][:, :, 2, :h11.shape[2] - Yh[i].size(3), :h11.shape[3] - Yh[i].size(4)] = h11[:, :, Yh[i].size(3):, Yh[i].size(4):] 
        # print(Yl,Yh[1])
        # Yl = Yl.cuda(device_id0)
        temp = ifm((Yl, [Yh[1]]))
        recons2 = ifm((temp, [Yh[0]]))
        recons2 = rearrange(recons2, "b c h w -> b h w c").contiguous()
        # # time4 = time.time()
        # # print(time4 - time3,'inverse wavelet')



        x = self.ln_1(input)
        # print(x.shape,'xshape')
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        # time5 = time.time()
        # print(time5 - time4,'2D Scan')

        input_freq = torch.fft.rfft2(prepare)+1e-8
        mag = torch.abs(input_freq)
        pha = torch.angle(input_freq)
        mag = self.block(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)+1e-8
        x_out = torch.fft.irfft2(x_out, s= tuple(x_size), norm='backward')+1e-8
        x_out = torch.abs(x_out)+1e-8
        x_out = rearrange(x_out, "b c h w -> b h w c").contiguous()

        
        # x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        
        # x1 = x + self.linear1(x_out)
        # x_out1 = x_out + self.linear2(x)

        # input_freq1 = torch.fft.rfft2(rearrange(x_out1, "b h w c -> b c h w").contiguous())+1e-8
        # mag1 = torch.abs(input_freq1)
        # pha1= torch.angle(input_freq1)
        # mag1 = self.block(mag1)
        # real1 = mag1 * torch.cos(pha1)
        # imag1 = mag1 * torch.sin(pha1)
        # x_out1 = torch.complex(real1, imag1)+1e-8
        # x_out1 = torch.fft.irfft2(x_out1, s= tuple(x_size), norm='backward')+1e-8
        # x_out1 = torch.abs(x_out1)+1e-8
        # x_out1 = rearrange(x_out1, "b c h w -> b h w c").contiguous()

        # x2 = self.ln_11(x1)
        # x2 = x1*self.skip_scale1 + self.drop_path1(self.self_attention1(x2))

        x = x.view(B, -1, C).contiguous()
        x_out = x_out.view(B, -1, C).contiguous()

        # wave trans
        x_dwt = recons2.view(B, -1, C).contiguous()
        
        # # print(x.shape,x_dwt.shape)
        

        # wave trans. The shapes may not match slightly due to the wavelet transform
        if x.shape != x_dwt.shape:
            x_dwt = x_dwt[:,:x.shape[1],:]

        # # wave trans
        x_final = torch.cat((x,x_out,x_dwt),2)
        x_final = self.linear_out(x_final)


        # time6 = time.time()
        # print(time6 - time5,'last')
        # print(time6 - time0,'all')
        # print((time4 - time0)/(time6 - time0),(time6 - time4)/ (time6 - time0))
        return x_final # B,HW,C

class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        # only used to run previous version
        if forward_type.startswith("v0"):
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
            return
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
            return
        else:
            self.__initv2__(**kwargs)
            return

    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        self.out_norm_shape = "v1"
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            class SoftmaxSpatial(nn.Softmax):
                def forward(self, x: torch.Tensor):
                    B, C, H, W = x.shape
                    return super().forward(x.view(B, C, -1)).view(B, C, H, W)
            self.out_norm = SoftmaxSpatial(dim=-1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        elif channel_first:
            self.out_norm = LayerNorm2d(d_inner)
        else:
            self.out_norm_shape = "v0"
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            ),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        # self.forward_core = FORWARD_TYPES.get(forward_type, None)
        self.forward_core = FORWARD_TYPES.get('v31d', None)  # hardcode
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    
    def forward_corev2(self, x: torch.Tensor, cross_selective_scan=cross_selective_scan, **kwargs):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")

        return cross_selective_scan(
            x, x_proj_weight, None, dt_projs_weight, dt_projs_bias,
            A_logs, Ds, delta_softplus=True,
            out_norm=out_norm,
            channel_first=self.channel_first,
            out_norm_shape=out_norm_shape,
            **kwargs,
        )

    def forwardv2(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        y = self.forward_core(x)

        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

########################## end #################################

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=12):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)
class CMF_Mamba(nn.Module):
    def __init__(self, dim):
        super(CMF_Mamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    
    def forward(self,ms,pan):
        
        ms = self.norm1(ms)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
        B,HW,C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, int(pow(HW,0.5)), int(pow(HW,0.5)))
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms

#________________________下面的是可能会用上,上面的是最近搭的最基础的框架________






def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms, pan):
        b, c, h, w = ms.shape

        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, ms,pan):
        ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim,bimamba_type=None)
        self.panencoder = Mamba(dim,bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, ms,pan
                ,ms_residual,pan_residual):
        # ms (B,N,C)
        #pan (B,N,C)
        ms_residual = ms+ms_residual
        pan_residual = pan+pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B,N,C = ms.shape
        ms_first_half = ms[:, :, :C//2]
        pan_first_half = pan[:, :, :C//2]
        ms_swap= torch.cat([pan_first_half,ms[:,:,C//2:]],dim=2)
        pan_swap= torch.cat([ms_first_half,pan[:,:,C//2:]],dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap,pan_swap,ms_residual,pan_residual
class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
        B,HW,C = global_f.shape
#         ms = global_f.transpose(1, 2).view(B, C, 128*8, 128*8)
        ms = global_f.transpose(1, 2).view(B, C, 128, 128)
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms,ms_resi
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi

if __name__ == '__main__':
    net = Net().to(device_id0)
    ms = torch.randn([1, 4, 32, 32]).to(device_id0)
    pan = torch.randn([1, 1, 128, 128]).to(device_id0)
    out = net(ms=ms, _=None, pan=pan)
    print(out.shape)
    