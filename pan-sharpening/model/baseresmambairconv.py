import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
from freqmamba import VSSBlock1
import selective_scan_cuda_oflex
from attentionfusion import AttentionFusionBlock
from CMMamba.Cross_Model_Mamba import CrossMamba_
from CMMamba.Cross_Model_Mamba import PatchEmbed
import torchvision.utils as vutils
from VMamba.classification.models.vmamba import VSSBlock 
sys.path.append('/media/user/scy/Pansharpening-main/pan-sharpening/model/MambaIR/basicsr')
sys.path.append('/media/user/scy/Pansharpening-main/pan-sharpening/model/MambaIR')
from MambaIR.analysis.model_zoo.mambaIR import MambaIR, buildMambaIR

from functools import partial
from typing import Optional, Callable, Any
from pytorch_wavelets import DWTForward, DWTInverse 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat


device_id0 = torch.device('cuda:0')

class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        # self.norm = nn.LayerNorm(dim,'with_bias')
        self.norm = nn.LayerNorm(dim,eps=1e-5)
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)

class Net(nn.Module):
    # def __init__(self, spectral_num=None, criterion=None, channel=64,args=None, **kwargs):
    def __init__(self, num_channels=None,base_filter=None,channel=64,args=None, **kwargs):
        super(Net, self).__init__()

        # self.criterion = criterion

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        '''PNN的卷积'''
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=channel, kernel_size=9, stride=1,padding = 4,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=5, stride=1,padding = 2,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=5, stride=1,padding = 2,
                               bias=True)
        '''MsConcatPan卷积回 MS_shape '''
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=3, stride=1,padding = 1,
                               bias=True)
        '''MS_fConcatPan_f 3x3卷积+1x1卷积回 MS_shape '''
        self.conv5 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1,padding = 1,
                               bias=True)
        self.conv6 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=1, stride=1,padding = 0,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv3x3 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, padding=1)
        self.conv3x3_24 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, padding=1)
        self.conv89_24 = nn.Conv2d(in_channels=89, out_channels=32, kernel_size=3, padding=1)
        self.conv112_24 = nn.Conv2d(in_channels=112, out_channels=32, kernel_size=3, padding=1)
        self.conv20_24 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, padding=1)
        self.conv192_24 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, padding=1)
        self.conv400_24 = nn.Conv2d(in_channels=400, out_channels=32, kernel_size=3, padding=1)
        # 1×1 卷积层，输入通道数为 32，输出通道数为 4
        self.conv1x1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)

        #VMamba input require
        self.feature_extraction = nn.Sequential(*[VSSBlock(5) for i in range(8)])
        #Mamba
        self.total_feature_extraction = nn.Sequential(*[SingleMambaBlock(5) for i in range(8)])

        self.rawD_to_token1 = PatchEmbed(in_chans=24, embed_dim=24, patch_size=1, stride=1)
        self.rgb_to_token1 = PatchEmbed(in_chans=24, embed_dim=24, patch_size=1, stride=1)
        self.deep_fusion1 = CrossMamba_(24)  # todo 跨模态融合模块
        #MambaIR 
        self.ms_feature_extraction = nn.Sequential(*[MambaIR(embed_dim = 32,img_size=128,in_chans=4,depths=(1, 1, 1, 1, 1, 1),upscale=1) for i in range(1)])
        self.pan_feature_extraction = nn.Sequential(*[MambaIR(embed_dim = 32,img_size=128,in_chans=1,depths=(1, 1, 1, 1, 1, 1),upscale=1) for i in range(1)])

        #传进去x,shape
        self.vssblock=VSSBlock1(hidden_dim=24, drop_path=0.1, d_state=16, expand=2.0)
        self.vssblock2 = VSSBlock1(hidden_dim=24, drop_path=0.1, d_state=16, expand=2.0)
        self.conv4_24 = nn.Conv2d(4, 24, kernel_size=1)
        self.conv4_96 = nn.Conv2d(4, 96, kernel_size=1)
        self.conv24_4 = nn.Conv2d(24, 4, kernel_size=1)
        self.conv96_4 = nn.Conv2d(96, 4, kernel_size=1)
        self.conv1_24 = nn.Conv2d(1, 24, kernel_size=1)
        self.conv1_96 = nn.Conv2d(1, 96, kernel_size=1)
        self.conv24_1=nn.Conv2d(24, 1, kernel_size=1)
        self.conv96_1 = nn.Conv2d(96, 1, kernel_size=1)


    

    def forward(self,lms,_,pan):
        att  = AttentionFusionBlock(
            img_size_uav=128,
            img_size_satellite=64,
            patch_size=8,
            dropout_rate=0.1,
            input_ndim=256,
            mid_ndim=[512, 256],
            attention_layer_num=12
        )
        att = att.to(device_id0)
        # x = cat(lms,pan)
        #print("lms",lms.shape) #4,4,32,32
        #print("pan",pan.shape)# 4,1,128,128
        ms_bic = F.interpolate(lms,scale_factor=4) # ms_bic 4,4,128,128
        #print("ms_bic", ms_bic.shape)
        identity = ms_bic #保留原始信息
        B, C, H, W = ms_bic.shape
        

        Bp, Cp, Hp, Wp = pan.shape

        '''
        baseline 
        x_size = (H, W)

        ms_bic1 = self.ms_feature_extraction(ms_bic)  # ms_bic 4,4,128,128
        ms_bic1z = self.ms_feature_extraction(ms_bic1)  # torch.Size([2, 16384, 24])
        pan1 = self.pan_feature_extraction(pan)  # pan 4,1,128,128
        pan1z = self.pan_feature_extraction(pan1)
        x1 = torch.cat([ms_bic1z, pan1z], dim=1)  # torch.Size([1, 5, 128, 128])


        ms_bic2 = self.ms_feature_extraction(ms_bic1z)  # ms_bic 4,4,128,128
        ms_bic2z = self.ms_feature_extraction(ms_bic2)
        pan2 = self.pan_feature_extraction(pan1z)
        pan2z = self.pan_feature_extraction(pan2)
        x2 = torch.cat([ms_bic2z, pan2z, x1], dim=1)  # torch.Size([1, 5, 128, 128])


        ms_bic3 = self.ms_feature_extraction(ms_bic2z)  # ms_bic 4,4,128,128
        ms_bic3 = self.ms_feature_extraction(ms_bic3)
        ms_bic3z = self.ms_feature_extraction(ms_bic3)
        pan3 = self.pan_feature_extraction(pan2z)
        pan3 = self.pan_feature_extraction(pan3)
        pan3z = self.pan_feature_extraction(pan3)
        x3 = torch.cat([ms_bic3z, pan3z, x2], dim=1)  # torch.Size([1, 5, 128, 128])

        ms_bic4 = self.ms_feature_extraction(ms_bic3z)  # ms_bic 4,4,128,128
        ms_bic4 = self.ms_feature_extraction(ms_bic4)
        ms_bic4z = self.ms_feature_extraction(ms_bic4)
        pan4 = self.pan_feature_extraction(pan3z)
        pan4 = self.pan_feature_extraction(pan4)
        pan4z = self.pan_feature_extraction(pan4)
        x4 = torch.cat([ms_bic4z, pan4z, x3], dim=1)  # torch.Size([1, 5, 128, 128])

        # print("x4",x4.shape)

        # x4=self.conv3x3_24(x4)
        # x4 = self.conv89_24(x4)
        #x4 = self.conv112_24(x4)
        x4 = self.conv20_24(x4)
        # x4 = self.conv400_24(x4)
        x4 = self.relu(x4)
        x4 = self.conv1x1(x4)
        output = x4 + identity
        -----------------------------------------
        fremamba+cat
        x_size = (H, W)
        ms_bic = self.conv4_24(ms_bic)  # ([2, 24, 128, 128])
        ms_bic = rearrange(ms_bic, "b c h w -> b (h w) c").contiguous()  # torch.Size([2, 16384, 24])
        ms_bic1 = self.vssblock(ms_bic, x_size)  # ms_bic 4,4,128,128
        ms_bic1 = self.vssblock(ms_bic1, x_size)  # torch.Size([2, 16384, 24])
        ms_bic1 = rearrange(ms_bic1, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        ms_bic1z=ms_bic1
        pan = self.conv1_24(pan)
        pan = rearrange(pan, "b c h w -> b (h w) c").contiguous()  # torch.Size([2, 16384, 1])
        pan1 = self.vssblock(pan, x_size)  # pan 4,1,128,128
        pan1 = self.vssblock(pan1, x_size)
        pan1 = rearrange(pan1, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        pan1z=pan1
        # ms_bic1 = ms_bic1.reshape(B, -1,  24)
        # pan1 = pan1.reshape(B, -1 , 24)
        pan1 = self.rawD_to_token1(pan1)
        ms_bic1t = self.rgb_to_token1(ms_bic1)
        ms_bic_att = self.deep_fusion1(ms_bic1t, pan1)
        ms_bic_att = self.conv24_4(ms_bic_att)
        dim = 24
        ms_bic1 = ms_bic1.reshape(B, dim, 128, 128)
        pan1 = pan1.reshape(B, dim, 128, 128)
        # pan1=self.conv24_1(pan1)
        ms_bic1a = self.conv24_4(ms_bic1)
        #print("ms_bic1a",ms_bic1a.shape)
        #print("ms_bic1",ms_bic1.shape)
        x1 = torch.cat([ms_bic1z, pan1z], dim=1)  # torch.Size([1, 5, 128, 128])

        ms_bic2 = rearrange(ms_bic1, "b c h w -> b (h w) c").contiguous()
        ms_bic2 = self.vssblock(ms_bic2, x_size)  # ms_bic 4,4,128,128
        ms_bic2 = self.vssblock(ms_bic2, x_size)
        ms_bic2 = rearrange(ms_bic2, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 48, 128, 128])
        ms_bic2z=ms_bic2
        # ms_bic2 = self.conv24_4(ms_bic2)  # torch.Size([2, 4, 128, 128])
        # pan2 = self.conv1_24(pan1)
        pan2 = rearrange(pan1, "b c h w -> b (h w) c").contiguous()  # 5,24,128,128->torch.Size([5, 16384, 24])
        pan2 = self.vssblock(pan2, x_size)
        pan2 = self.vssblock(pan2, x_size)
        pan2 = rearrange(pan2, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        pan2z = pan2
        pan2 = self.rawD_to_token1(pan2)
        ms_bic2t = self.rgb_to_token1(ms_bic2)
        ms_bic_att2 = self.deep_fusion1(ms_bic2t, pan2)
        ms_bic_att2 = self.conv24_4(ms_bic_att2)
        ms_bic2 = ms_bic2.reshape(B, dim, 128, 128)
        pan2 = pan2.reshape(B, dim, 128, 128)
        # pan2 = self.conv24_1(pan2)
        ms_bic2a = self.conv24_4(ms_bic2)
        x2 = torch.cat([ms_bic2z, pan2z, x1], dim=1)  # torch.Size([1, 5, 128, 128])

        ms_bic3 = rearrange(ms_bic2, "b c h w -> b (h w) c").contiguous()
        ms_bic3 = self.vssblock(ms_bic3, x_size)  # ms_bic 4,4,128,128
        ms_bic3 = self.vssblock(ms_bic3, x_size)
        ms_bic3 = self.vssblock(ms_bic3, x_size)
        ms_bic3 = rearrange(ms_bic3, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 48, 128, 128])
        ms_bic3z = ms_bic3
        # ms_bic3 = self.conv24_4(ms_bic3)  # torch.Size([2, 4, 128, 128])
        # pan3 = self.conv1_24(pan2)
        pan3 = rearrange(pan2, "b c h w -> b (h w) c").contiguous()  # torch.Size([2, 16384, 1])
        pan3 = self.vssblock(pan3, x_size)
        pan3 = self.vssblock(pan3, x_size)
        pan3= self.vssblock(pan3, x_size)
        pan3 = rearrange(pan3, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        pan3z = pan3
        # pan3 = self.conv24_1(pan3)  # torch.Size([2, 1, 128, 128])
        pan3 = self.rawD_to_token1(pan3)
        ms_bic3t = self.rgb_to_token1(ms_bic3)
        ms_bic_att3 = self.deep_fusion1(ms_bic3t, pan3)
        ms_bic_att3 = self.conv24_4(ms_bic_att3)
        ms_bic3 = ms_bic3.reshape(B, dim, 128, 128)
        pan3 = pan3.reshape(B, dim, 128, 128)
        # pan3 = self.conv24_1(pan3)
        ms_bic3a = self.conv24_4(ms_bic3)
        x3 = torch.cat([ms_bic3z, pan3z, x2], dim=1)  # torch.Size([1, 5, 128, 128])

        ms_bic4 = rearrange(ms_bic3, "b c h w -> b (h w) c").contiguous()
        ms_bic4 = self.vssblock(ms_bic4, x_size)  # ms_bic 4,4,128,128
        ms_bic4 = self.vssblock(ms_bic4, x_size)
        ms_bic4 = self.vssblock(ms_bic4, x_size)
        ms_bic4 = rearrange(ms_bic4, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 48, 128, 128])
        ms_bic4z = ms_bic4
        # ms_bic4 = self.conv24_4(ms_bic4)  # torch.Size([2, 4, 128, 128])
        # pan4 = self.conv1_24(pan3)
        pan4 = rearrange(pan3, "b c h w -> b (h w) c").contiguous()  # torch.Size([2, 16384, 1])
        pan4 = self.vssblock(pan4, x_size)
        pan4 = self.vssblock(pan4, x_size)
        pan4 = self.vssblock(pan4, x_size)
        pan4 = rearrange(pan4, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        pan4z = pan4
        # pan3 = self.conv24_1(pan3)  # torch.Size([2, 1, 128, 128])
        pan4 = self.rawD_to_token1(pan4)
        ms_bic4t = self.rgb_to_token1(ms_bic4)
        ms_bic_att4 = self.deep_fusion1(ms_bic4t, pan4)
        ms_bic_att4 = self.conv24_4(ms_bic_att4)
        ms_bic4 = ms_bic4.reshape(B, dim, 128, 128)
        pan4 = pan4.reshape(B, dim, 128, 128)
        # pan4 = self.conv96_1(pan4)
        ms_bic4a = self.conv24_4(ms_bic4)
        x4 = torch.cat([ms_bic4z, pan4z, x3], dim=1)  # torch.Size([1, 5, 128, 128])

        #print("x4",x4.shape)

        # x4=self.conv3x3_24(x4)
        # x4 = self.conv89_24(x4)
        #x4 = self.conv112_24(x4)
        x4 = self.conv192_24(x4)
        # x4 = self.conv400_24(x4)
        x4 = self.relu(x4)
        x4 = self.conv1x1(x4)
        output = x4 + identity
        # output = x4
        '''
        x_size = (H, W)
        ms_bic = self.conv4_24(ms_bic)  # ([2, 24, 128, 128])
        ms_bic = rearrange(ms_bic, "b c h w -> b (h w) c").contiguous()  # torch.Size([2, 16384, 24])
        ms_bic1 = self.vssblock(ms_bic, x_size)  # ms_bic 4,4,128,128
        ms_bic1 = self.vssblock(ms_bic1, x_size)  # torch.Size([2, 16384, 24])
        ms_bic1 = rearrange(ms_bic1, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        pan = self.conv1_24(pan)
        pan = rearrange(pan, "b c h w -> b (h w) c").contiguous()  # torch.Size([2, 16384, 1])
        pan1 = self.vssblock(pan, x_size)  # pan 4,1,128,128
        pan1 = self.vssblock(pan1, x_size)
        pan1 = rearrange(pan1, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        # ms_bic1 = ms_bic1.reshape(B, -1,  24)
        # pan1 = pan1.reshape(B, -1 , 24)
        pan1 = self.rawD_to_token1(pan1)
        ms_bic1t = self.rgb_to_token1(ms_bic1)
        ms_bic_att = self.deep_fusion1(ms_bic1t, pan1)
        ms_bic_att = self.conv24_4(ms_bic_att)
        dim = 24
        ms_bic1 = ms_bic1.reshape(B, dim, 128, 128)
        pan1 = pan1.reshape(B, dim, 128, 128)
        # pan1=self.conv24_1(pan1)
        x1 = torch.cat([ms_bic_att, pan1], dim=1)  # torch.Size([1, 5, 128, 128])

        ms_bic2 = rearrange(ms_bic1, "b c h w -> b (h w) c").contiguous()
        ms_bic2 = self.vssblock(ms_bic2, x_size)  # ms_bic 4,4,128,128
        ms_bic2 = self.vssblock(ms_bic2, x_size)
        ms_bic2 = rearrange(ms_bic2, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 48, 128, 128])
        # ms_bic2 = self.conv24_4(ms_bic2)  # torch.Size([2, 4, 128, 128])
        # pan2 = self.conv1_24(pan1)
        pan2 = rearrange(pan1, "b c h w -> b (h w) c").contiguous()  # 5,24,128,128->torch.Size([5, 16384, 24])
        pan2 = self.vssblock(pan2, x_size)
        pan2 = self.vssblock(pan2, x_size)
        pan2 = rearrange(pan2, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        pan2 = self.rawD_to_token1(pan2)
        ms_bic2t = self.rgb_to_token1(ms_bic2)
        ms_bic_att2 = self.deep_fusion1(ms_bic2t, pan2)
        ms_bic_att2 = self.conv24_4(ms_bic_att2)
        ms_bic2 = ms_bic2.reshape(B, dim, 128, 128)
        pan2 = pan2.reshape(B, dim, 128, 128)
        # pan2 = self.conv24_1(pan2)
        x2 = torch.cat([ms_bic_att2, pan2, x1], dim=1)  # torch.Size([1, 5, 128, 128])

        ms_bic3 = rearrange(ms_bic2, "b c h w -> b (h w) c").contiguous()
        ms_bic3 = self.vssblock(ms_bic3, x_size)  # ms_bic 4,4,128,128
        ms_bic3 = self.vssblock(ms_bic3, x_size)
        ms_bic3 = self.vssblock(ms_bic3, x_size)
        ms_bic3 = rearrange(ms_bic3, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 48, 128, 128])
        # ms_bic3 = self.conv24_4(ms_bic3)  # torch.Size([2, 4, 128, 128])
        # pan3 = self.conv1_24(pan2)
        pan3 = rearrange(pan2, "b c h w -> b (h w) c").contiguous()  # torch.Size([2, 16384, 1])
        pan3 = self.vssblock(pan3, x_size)
        pan3 = self.vssblock(pan3, x_size)
        pan3 = self.vssblock(pan3, x_size)
        pan3 = rearrange(pan3, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        # pan3 = self.conv24_1(pan3)  # torch.Size([2, 1, 128, 128])
        pan3 = self.rawD_to_token1(pan3)
        ms_bic3t = self.rgb_to_token1(ms_bic3)
        ms_bic_att3 = self.deep_fusion1(ms_bic3t, pan3)
        ms_bic_att3 = self.conv24_4(ms_bic_att3)
        ms_bic3 = ms_bic3.reshape(B, dim, 128, 128)
        pan3 = pan3.reshape(B, dim, 128, 128)
        # pan3 = self.conv24_1(pan3)
        x3 = torch.cat([ms_bic_att3, pan3, x2], dim=1)  # torch.Size([1, 5, 128, 128])

        ms_bic4 = rearrange(ms_bic3, "b c h w -> b (h w) c").contiguous()
        ms_bic4 = self.vssblock(ms_bic4, x_size)  # ms_bic 4,4,128,128
        ms_bic4 = self.vssblock(ms_bic4, x_size)
        ms_bic4 = self.vssblock(ms_bic4, x_size)
        ms_bic4 = rearrange(ms_bic4, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 48, 128, 128])
        # ms_bic4 = self.conv24_4(ms_bic4)  # torch.Size([2, 4, 128, 128])
        # pan4 = self.conv1_24(pan3)
        pan4 = rearrange(pan3, "b c h w -> b (h w) c").contiguous()  # torch.Size([2, 16384, 1])
        pan4 = self.vssblock(pan4, x_size)
        pan4 = self.vssblock(pan4, x_size)
        pan4 = self.vssblock(pan4, x_size)
        pan4 = rearrange(pan4, 'b (h w) c -> b c h w', h=128, w=128).contiguous()  # torch.Size([2, 24, 128, 128])
        # pan3 = self.conv24_1(pan3)  # torch.Size([2, 1, 128, 128])
        pan4 = self.rawD_to_token1(pan4)
        ms_bic4t = self.rgb_to_token1(ms_bic4)
        ms_bic_att4 = self.deep_fusion1(ms_bic4t, pan4)
        ms_bic_att4 = self.conv24_4(ms_bic_att4)
        ms_bic4 = ms_bic4.reshape(B, dim, 128, 128)
        pan4 = pan4.reshape(B, dim, 128, 128)
        # pan4 = self.conv96_1(pan4)
        x4 = torch.cat([ms_bic_att4, pan4, x3], dim=1)  # torch.Size([1, 5, 128, 128])

        # print("x4",x4.shape)

        # x4=self.conv3x3_24(x4)
        # x4 = self.conv89_24(x4)
        x4 = self.conv112_24(x4)
        # x4 = self.conv400_24(x4)
        x4 = self.relu(x4)
        x4 = self.conv1x1(x4)
        output = x4 + identity
        
       
        # # '''Mamba require B,L,C 
        # x = x.flatten(2).transpose(1, 2) # b,c,h,w -> b,L,c
        # residual_total_f = 0
        # x,residual_total_f = self.total_feature_extraction([x,residual_total_f])
        # x = x.transpose(1, 2).view(B,C,H,W) 
        # # '''


        '''8VMamba require (B, H, W, C) '''
        # x = self.feature_extraction(x)

        # x = x.permute(0, 3, 1, 2).contiguous()#b,h,w,c-> b,c,h,w VMamba

        # x = x.transpose(1, 2).view(B, C, H, W)
        # x = x.unflatten(2, (H, W))
        
        # x = self.ms_to_token(x)
        # print(x.shape)

        
        '''PNN require require (B, C, H, W) '''
        # rs = self.relu(self.conv1(x))
        # rs = self.relu(self.conv2(rs))
        # output = self.conv3(rs)
        

        return output




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


class PatchEmbed1(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # print(x.shape)
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        #x = self.proj(x)
        # print(x.shape)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # print(x.shape)
        # x = self.norm(x)
        return x



if __name__ == '__main__':
    net = Net().to(device_id0)
    ms = torch.randn([1, 4, 32, 32]).to(device_id0)
    pan = torch.randn([1, 1, 128, 128]).to(device_id0)
    out = net(lms=ms, _=None, pan=pan)
    print(out.shape)

