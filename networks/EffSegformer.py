import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        # print('input in DWConv: {}'.format(x.shape))
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out

class MixFFN_skip_fuse(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out
         
class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MixD_FFN(nn.Module):
    def __init__(self, c1, c2, fuse_mode = "add"):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode=="add" else nn.Linear(c2*2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x):
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax+self.fc1(x)) if self.fuse_mode=="add" else self.act(torch.cat([ax, self.fc1(x)],2))
        out = self.fc2(ax) 
        return out

class OverlapPatchEmbeddings_fuse_padding(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, dilation=1, in_ch=3, dim=768, p2size=28):
        super().__init__()
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding, dilation)
        self.norm = nn.LayerNorm(dim)
        self.p2size = p2size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        px = self.proj(x)   
        _, _, H, W = px.shape
        H1=H
        H2=self.p2size
        H=H2
        W=H2
        if H1 != H2:
            p2d = (int((H2-H1)/2),int((H2-H1)/2),int((H2-H1)/2),int((H2-H1)/2))
            px = F.pad(px, p2d, "constant", 0)

        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W     


class OverlapPatchEmbeddings_fuse(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, dilation=1, in_ch=3, dim=768):
        super().__init__()
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding, dilation)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        px = self.proj(x)   
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W     

class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W  


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))



# # proj_dilation1 = nn.Conv2d(input_dim,in_dim[i_stage],kernel_size = patch_sizes1[i_stage], stride = strides[i_stage],padding=dil_padding_sizes1[i_stage], dilation=2)
# conv3_3_1 =  nn.Conv2d(input_dim, in_dim[i_stage], kernel_size=3, padding =1)
# conv3_3_2 =  nn.Conv2d(input_dim, in_dim[i_stage], kernel_size=3, padding =1)
#https://github.com/Cassieyy/MultiResUnet3D/blob/main/MultiResUnet3D.py

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, act='relu'):
        # print(ch_out)
        super(conv_block,self).__init__()
        if act == None:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding),
                nn.BatchNorm2d(ch_out)
            )
        elif act == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(ch_out)
                # yiwei here change ReLU and bn
            )
        elif act == 'sigmoid':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.Sigmoid()
            )

    def forward(self,x):
        x = self.conv(x)
        return x
    


class res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(res_block,self).__init__()
        self.res = conv_block(ch_in,ch_out,1,1,0,None)
        self.main = conv_block(ch_in,ch_out)
        self.bn = nn.BatchNorm2d(ch_in)
    def forward(self,x):
        res_x = self.res(x)

        main_x = self.main(x)
        out = res_x.add(main_x)
        out = nn.ReLU(inplace=True)(out)
        out = self.bn(out)    
        # yiwei: here change ReLU and bn order  
        # print(out.shape[1], type(out.shape[1]))
        # assert 1>3
        
        
        return out
    

# unchange->not often used
# class MultiResBlock_1357(nn.Module):
#     def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
#         super(MultiResBlock_1357,self).__init__()
#         self.W = alpha * U
#         self.one_ch = conv_block(in_ch, 1)
# #         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
#         self.residual_layer = conv_block(1, self.W)
# #         self.conv3x3 = conv_block(1, int(self.W*0.167))
# #         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
# #         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
#         self.conv3x3 = conv_block(1, int(self.W))
#         self.conv5x5 = conv_block(int(self.W), int(self.W))
#         self.conv7x7 = conv_block(int(self.W), self.W)
#         self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
#         self.relu = nn.ReLU(inplace=True)
# #         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
# #         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
# #         self.batchnorm_1 = nn.BatchNorm2d(self.W)
# #         self.batchnorm_2 = nn.BatchNorm2d(self.W)
#         self.norm = nn.LayerNorm(self.W)
        
#     def forward(self, x):
#         out = []
#         # print(x.shape) 
#         # print("\n W=alpha*U :{}\n".format(self.W))
#         x = self.one_ch(x) 
#         res = self.residual_layer(x)
#         res_out = self.maxpool(res)
#         res_out = (res_out.flatten(2)).transpose(1,2)
#         out.append(res_out)
#         # print("\n res:{}\n".format(res_out.shape))
        
#         sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         # print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         # print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         # print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
#         all_t = torch.cat((out[0], out[1], out[2],out[3]), 1)
#         all_t = self.norm(all_t)
#         # print("\n cat_together:{}\n".format(all_t.shape))
# #         all_t_b = self.batchnorm_1(all_t)
# #         out = all_t_b.add(res)
# #         out = self.relu(out)
# #         out = self.batchnorm_2(out)      
        
#         return all_t
    
    
# class MultiResBlock_135(nn.Module):
#     def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
#         super(MultiResBlock_135,self).__init__()
#         self.W = alpha * U
#         # self.one_ch = conv_block(in_ch, in_ch)
# #         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
#         self.residual_layer = conv_block(in_ch, self.W)
# #         self.conv3x3 = conv_block(1, int(self.W*0.167))
# #         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
# #         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
#         self.conv3x3 = conv_block(in_ch, int(self.W))
#         self.conv5x5 = conv_block(int(self.W), int(self.W))
#         self.conv7x7 = conv_block(int(self.W), int(self.W))
#         self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
#         self.relu = nn.ReLU(inplace=True)
# #         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
# #         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
# #         self.batchnorm_1 = nn.BatchNorm2d(self.W)
# #         self.batchnorm_2 = nn.BatchNorm2d(self.W)
#         self.norm = nn.BatchNorm2d(self.W)
        
#     def forward(self, x):
#         out = []
#         # print(x.shape) 
#         # print("\n W=alpha*U :{}\n".format(self.W))
#         # x = self.one_ch(x) 
#         res = self.residual_layer(x)
#         res_out = self.maxpool(res)
#         res_out = (res_out.flatten(2)).transpose(1,2)
#         out.append(res_out)
#         # print("\n res:{}\n".format(res_out.shape))
        
#         sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         # print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         # print("\n out_5*5:{}\n".format(obo_out.shape))
        
# #         cbc = self.conv7x7(obo)
# #         cbc_out = self.maxpool(cbc)
# #         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
# #         out.append(cbc_out)
# #         print("\n out_7*7:{}\n".format(cbc_out.shape))
        
        
#         all_t = torch.cat(out, 1)
#         all_t = self.norm(all_t)
#         # print("\n cat_together:{}\n".format(all_t.shape))
# #         all_t_b = self.batchnorm_1(all_t)
# #         out = all_t_b.add(res)
# #         out = self.relu(out)
# #         out = self.batchnorm_2(out)      
        
#         return all_t

# class MultiResBlock_157(nn.Module):
#     def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
#         super(MultiResBlock_157,self).__init__()
#         self.W = alpha * U
#         self.one_ch = conv_block(in_ch, 1)
# #         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
# #         self.conv3x3 = conv_block(1, int(self.W*0.167))
# #         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
# #         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
#         self.conv3x3 = conv_block(1, int(self.W))
#         self.conv5x5 = conv_block(int(self.W), int(self.W))
#         self.conv7x7 = conv_block(int(self.W), self.W)
#         self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
#         self.relu = nn.ReLU(inplace=True)
# #         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
# #         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
# #         self.batchnorm_1 = nn.BatchNorm2d(self.W)
# #         self.batchnorm_2 = nn.BatchNorm2d(self.W)
#         self.norm = nn.LayerNorm(self.W)
        
#     def forward(self, x):
#         out = []
#         # print(x.shape) 
#         # print("\n W=alpha*U :{}\n".format(self.W))
#         x = self.one_ch(x) 
#         res = self.residual_layer(x)
#         res_out = self.maxpool(res)
#         res_out = (res_out.flatten(2)).transpose(1,2)
#         out.append(res_out)
#         # print("\n res:{}\n".format(res_out.shape))
        
#         sbs = self.conv3x3(x)
# #         sbs_out = self.maxpool(sbs)
# #         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
# #         print("\n out_3*3:{}\n".format(sbs_out.shape))
# #         out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         # print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         # print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
#         all_t = torch.cat(out, 1)
#         all_t = self.norm(all_t)
#         # print("\n cat_together:{}\n".format(all_t.shape))
# #         all_t_b = self.batchnorm_1(all_t)
# #         out = all_t_b.add(res)
# #         out = self.relu(out)
# #         out = self.batchnorm_2(out)      
        
#         return all_t
    

class MultiResBlock_15(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_15,self).__init__()
        self.W = alpha * U
        # self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(in_ch, self.W, 1, 1, 0, act=None)

        self.conv3x3 = conv_block(in_ch, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        # print(x.shape) 
        # print("\n W=alpha*U :{}\n".format(self.W))
        
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        # print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
        obo = self.conv5x5(sbs)
        obo_out = self.maxpool(obo)
        obo_out = (obo_out.flatten(2)).transpose(1,2)
        out.append(obo_out)
        # print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        all_t = torch.cat(out, 1)
        
        all_t = self.norm(all_t)
        # print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t

    
class MultiResBlock_13(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_13,self).__init__()
        self.W = alpha * U
        # self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(in_ch, self.W, 1, 1, 0, act=None)
        self.conv3x3 = conv_block(in_ch, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        # print(x.shape) 
        # print("\n W=alpha*U :{}\n".format(self.W))
        # x = self.one_ch(x) 
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        # print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
        sbs_out = self.maxpool(sbs)
        sbs_out = (sbs_out.flatten(2)).transpose(1,2)
        # print("\n out_3*3:{}\n".format(sbs_out.shape))
        out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        all_t = torch.cat(out, 1)
        
        all_t = self.norm(all_t)
        # print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
    
    
class MultiResBlock_1(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_1,self).__init__()
        self.W = alpha * U
        # self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(in_ch, self.W, 1, 1, 0, act=None)
        self.conv3x3 = conv_block(in_ch, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        # print(x.shape) 
        # print("\n W=alpha*U :{}\n".format(self.W))
        # x = self.one_ch(x) 
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        # print("\n res:{}\n".format(res_out.shape))
        
#         sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        # print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
class MultiResBlock_3(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_3,self).__init__()
        self.W = alpha * U
        # self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(in_ch, self.W, 1, 1, 0, act=None)
        self.conv3x3 = conv_block(in_ch, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        # print(x.shape) 
        # print("\n W=alpha*U :{}\n".format(self.W))
        # x = self.one_ch(x) 
#         res = self.residual_layer(x)
#         res_out = self.maxpool(res)
#         res_out = (res_out.flatten(2)).transpose(1,2)
#         out.append(res_out)
#         print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
        sbs_out = self.maxpool(sbs)
        sbs_out = (sbs_out.flatten(2)).transpose(1,2)
        # print("\n out_3*3:{}\n".format(sbs_out.shape))
        out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        # print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
class MultiResBlock_5(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_5,self).__init__()
        self.W = alpha * U      
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(in_ch, self.W, 1, 1, 0, act=None)
        self.conv3x3 = conv_block(in_ch, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        # print(x.shape) 
        # print("\n W=alpha*U :{}\n".format(self.W))
        # x = self.one_ch(x) 
#         res = self.residual_layer(x)
#         res_out = self.maxpool(res)
#         res_out = (res_out.flatten(2)).transpose(1,2)
#         out.append(res_out)
#         print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
        obo = self.conv5x5(sbs)
        obo_out = self.maxpool(obo)
        obo_out = (obo_out.flatten(2)).transpose(1,2)
        out.append(obo_out)
        # print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        # print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
    
    
# class MultiResBlock_7(nn.Module):
#     def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
#         super(MultiResBlock_7,self).__init__()
#         self.W = alpha * U
#         self.one_ch = conv_block(in_ch, 1)
# #         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
#         self.residual_layer = conv_block(1, self.W)
# #         self.conv3x3 = conv_block(1, int(self.W*0.167))
# #         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
# #         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
#         self.conv3x3 = conv_block(1, int(self.W))
#         self.conv5x5 = conv_block(int(self.W), int(self.W))
#         self.conv7x7 = conv_block(int(self.W), self.W)
#         self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
#         self.relu = nn.ReLU(inplace=True)
# #         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
# #         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
# #         self.batchnorm_1 = nn.BatchNorm2d(self.W)
# #         self.batchnorm_2 = nn.BatchNorm2d(self.W)
#         self.norm = nn.LayerNorm(self.W)
        
#     def forward(self, x):
#         out = []
#         # print(x.shape) 
#         # print("\n W=alpha*U :{}\n".format(self.W))
#         x = self.one_ch(x) 
# #         res = self.residual_layer(x)
# #         res_out = self.maxpool(res)
# #         res_out = (res_out.flatten(2)).transpose(1,2)
# #         out.append(res_out)
# #         print("\n res:{}\n".format(res_out.shape))
        
#         sbs = self.conv3x3(x)
# #         sbs_out = self.maxpool(sbs)
# #         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
# #         print("\n out_3*3:{}\n".format(sbs_out.shape))
# #         out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
# #         obo_out = self.maxpool(obo)
# #         obo_out = (obo_out.flatten(2)).transpose(1,2)
# #         out.append(obo_out)
# #         print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         # print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
#         all_t = torch.cat(out, 1)
#         all_t = self.norm(all_t)
#         # print("\n cat_together:{}\n".format(all_t.shape))
# #         all_t_b = self.batchnorm_1(all_t)
# #         out = all_t_b.add(res)
# #         out = self.relu(out)
# #         out = self.batchnorm_2(out)      
        
#         return all_t