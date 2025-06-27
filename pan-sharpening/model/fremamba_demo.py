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
# sys.path.append('/root/Pan-Mamba/pan-sharpening/MambaIR/basicsr') 
# sys.path.append('/root/Pan-Mamba/pan-sharpening/MambaIR') 
# from MambaIR.analysis.model_zoo.mambaIR import MambaIR, buildMambaIR

from functools import partial
from typing import Optional, Callable
from pytorch_wavelets import DWTForward, DWTInverse 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from typing import Optional, Callable, Any

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
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=channel, kernel_size=9, stride=1,padding = 4,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=5, stride=1,padding = 2,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=5, stride=1,padding = 2,
                               bias=True)
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=3, stride=1,padding = 1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        # init_weights(self.conv1, self.conv2, self.conv3)
        # self.ms_encoder = nn.Sequential(nn.Conv2d(5,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        #Tokenization
        # self.ms_to_token = PatchEmbed(in_chans=5, embed_dim=5, patch_size=1, stride=1)
        # self.ms_feature_extraction_mambair = MambaIR(embed_dim = 5,img_size=128,in_chans=5,depths=(1,1,1),upscale=1)
       
        #VMamba
        # self.feature_extraction = VSSBlock(hidden_dim=5, drop_path=0.0, norm_layer=nn.LayerNorm, attn_drop_rate=0.0,**kwargs) 
        # self.feature_extraction_0 = VSSBlock(hidden_dim=5, drop_path=0.0, norm_layer=nn.LayerNorm, attn_drop_rate=0.0,**kwargs) 
        # self.feature_extraction_1 = VSSBlock(hidden_dim=5, drop_path=0.0, norm_layer=nn.LayerNorm, attn_drop_rate=0.0,**kwargs) 
        # self.feature_extraction_2 = VSSBlock(hidden_dim=5, drop_path=0.0, norm_layer=nn.LayerNorm, attn_drop_rate=0.0,**kwargs) 

        self.feature_extraction = nn.Sequential(*[VSSBlock(5) for i in range(8)])
        #Mamba
        self.total_feature_extraction = nn.Sequential(*[SingleMambaBlock(5) for i in range(8)])


    

    def forward(self,lms,_,pan):  # x = cat(lms,pan)
        ms_bic = F.interpolate(lms,scale_factor=4)

        x = torch.cat([ms_bic, pan], dim=1)
    
        # x = pan
        # x = self.ms_encoder(x) 
        # x = self.ms_feature_extraction_mambair(x)
        B, C, H, W = x.shape
        # x = x.flatten(2)
        # x = x.flatten(2).transpose(1, 2)
        # x = x.permute(0, 2, 3, 1).contiguous()#b,c,h,w -> b,h,w,c VMamba
        
        # '''Mamba
        x = x.flatten(2).transpose(1, 2) # b,c,h,w -> b,L,c
        residual_total_f = 0
        x,residual_total_f = self.total_feature_extraction([x,residual_total_f])
        x = x.transpose(1, 2).view(B,C,H,W) 
        # '''

        
        '''VMamba
        x = self.feature_extraction(x)
        x = self.feature_extraction_0(x)
        x = self.feature_extraction_1(x)
        x = self.feature_extraction_2(x)
        '''

        '''8VMamba'''
        # x = self.feature_extraction(x)

        # x = x.permute(0, 3, 1, 2).contiguous()#b,h,w,c-> b,c,h,w VMamba

        # x = x.transpose(1, 2).view(B, C, H, W)
        # x = x.unflatten(2, (H, W))
        
        # x = self.ms_to_token(x)
        # print(x.shape)
        '''PNN'''
        rs = self.relu(self.conv1(x))
        rs = self.relu(self.conv2(rs))
        output = self.conv3(rs)
        # output = x
        #print(lms.shape,ms_bic.shape,x.shape,rs.shape)
        # output = self.relu(self.conv4(x))

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

