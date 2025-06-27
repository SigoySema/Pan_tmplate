import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba

import selective_scan_cuda_oflex


from VMamba.classification.models.vmamba import VSSBlock 
sys.path.append('/root/Pan-Mamba/pan-sharpening/model/MambaIR/basicsr') 
sys.path.append('/root/Pan-Mamba/pan-sharpening/model/MambaIR') 
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

       
        

        #VMamba input require
        self.feature_extraction = nn.Sequential(*[VSSBlock(5) for i in range(8)])
        #Mamba
        self.total_feature_extraction = nn.Sequential(*[SingleMambaBlock(5) for i in range(8)])
        
        #MambaIR 
        self.ms_feature_extraction = nn.Sequential(*[MambaIR(embed_dim = 32,img_size=128,in_chans=4,depths=(1, 1, 1, 1, 1, 1),upscale=1) for i in range(1)])
        self.pan_feature_extraction = nn.Sequential(*[MambaIR(embed_dim = 32,img_size=128,in_chans=1,depths=(1, 1, 1, 1, 1, 1),upscale=1) for i in range(1)])
        #传进去x,shape


    

    def forward(self,lms,_,pan):  # x = cat(lms,pan)
        ms_bic = F.interpolate(lms,scale_factor=4)
        identity = ms_bic #保留原始信息
        B, C, H, W = ms_bic.shape
        

        Bp, Cp, Hp, Wp = pan.shape

        '''MambaIR require (B, C, H, W)'''
        ms_bic = self.ms_feature_extraction(ms_bic)
        pan = self.pan_feature_extraction(pan)
        x = torch.cat([ms_bic, pan], dim=1)
        x = self.relu(self.conv5(x))
        output = self.conv6(x)+identity
        
       
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


class PatchEmbed(nn.Module):
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

