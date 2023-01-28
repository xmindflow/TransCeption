from cmath import sqrt
from re import X
import torch
import torch.nn as nn
from torch import einsum
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from torchvision import models
from torch.nn import init
from torchinfo import summary
# from torchstat import stat

from functools import partial
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import DropPath, trunc_normal_


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

class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EfficientAttention(nn.Module):
    """
        input  -> x:[B, D, H, W]
        output ->   [B, D, H, W]
    
        in_channels:    int -> Embedding Dimension 
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
        
        Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """
    
    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

        
    def forward(self, input_):
        n, _, h, w = input_.size()
        # n, _,  = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
                        
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]            
            
            context = key @ value.transpose(1, 2) # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w) # n*dv            
            attended_values.append(attended_value)
                
        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class EfficientTransformerBlock(nn.Module):
    """
        Input  -> x (Size: (b, (H*W), d)), H, W
        Output -> (b, (H*W), d)
    """
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim,
                                       value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim*4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
        attn = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)
        
        tx = x + attn
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx   
    

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x): 
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)
        
        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x.clone())

        return x
    
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x
  

class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = EfficientTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = EfficientTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)
      
    def forward(self, x1, x2=None):
        if x2 is not None:# skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out

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


## MSViT modules

class DWConv2d_BN(nn.Module):
    """
    Depthwise Separable Conv
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
        norm_cfg="BN",
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, sqrt(2.0 / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act_layer=None,
        norm_cfg="BN",
    ):
        super().__init__()
        # self.add_module('c', torch.nn.Conv2d(
        #     a, b, ks, stride, pad, dilation, groups, bias=False))
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # Note that there is no bias due to BN
                # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x

class Conv3d_BN_concat(nn.Module):
    # input: attention list of length #path
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act_layer=None,
        norm_cfg="BN",
    ):
        super().__init__()
        # self.add_module('c', torch.nn.Conv2d(
        #     a, b, ks, stride, pad, dilation, groups, bias=False))
        # self.conv = nn.Conv2d(
        #     in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False
        # )
        self.bn = nn.BatchNorm2d(out_ch)

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # Note that there is no bias due to BN
                # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))

        # self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.interact_concat = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(4,1,1)),
            nn.ReLU()
        )

    def forward(self, x_list):
        
        b,c,h,w = x_list[0].shape
        out_3d = []
        for ip in range(len(x_list)):
            x = x_list[ip]
            x = x.unsqueeze_(dim=2)
            out_3d.append(x)
            # print(f'{ip} path extend to shape{x.shape}')

        x = torch.cat(out_3d, dim=2)
        print(f'after concat: {x.shape}')
        x = torch.squeeze(self.interact_concat(x), dim=2)
        print(f'after squeeze: {x.shape}')
        x = self.bn(x)

        return x

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.numpath = 3

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B  C  numPath H  W)
            returns :
                out : attention value + input feature
                attention: B X C X numPath X numPath
        """
        m_batchsize, C,numpath, height, width = x.size()
        proj_query = x.reshape(m_batchsize, C, numpath, -1) #b c 4 n
        # print(f'Shape of query: {proj_query.shape}')
        proj_key = x.reshape(m_batchsize, C,numpath, -1).permute(0, 1, 3, 2) #b c n 4
        # print(f'Shape of key: {proj_key.shape}') 
        energy = torch.matmul(proj_query, proj_key) # b c 4 4
        print(f'Shape of energy: {energy[0][0]}')
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy# The larger dependency of the channels, the lower energy_new value/ used to emphasizing the different information
        
        # min_energy_0 = torch.min(energy, -1, keepdim=True)[0].expand_as(energy)
        # energy_new = energy - min_energy_0
        print(f'energy_new: {energy_new[0][0]}')
        
        attention = self.softmax(energy_new)
        print(f'Shape of attention: {attention[0][0]}\n')
        proj_value = x.reshape(m_batchsize, C, numpath, -1)
        # print(f'Shape of proj_value: {proj_value.shape}')

        out = torch.matmul(attention, proj_value)# can be replace by torch.matmul
        out = out.reshape(m_batchsize, C, numpath, height, width)


#         logging.debug('cam device: {}, {}'.format(out.device, self.gamma.device))
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out


class CAM_Factorized_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim
        
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.numpath = 3
        self.num_heads = 8
        self.scale = (in_dim // self.num_heads) ** -0.5
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.proj = nn.Linear(in_dim, in_dim)
        crpe_window={3: 2, 5: 3, 7: 3}
        # self.crpe = shared_crpe
        self.crpe = ConvRelPosEnc(Ch=in_dim // self.num_heads, h=self.num_heads, window=crpe_window)

    def forward(self, x):
        B, C, numpath, height, width = x.size()
        x1 = x.reshape(B,C,-1).permute(0,2,1)
        B,N,C = x1.size()
        # print(f'B:{B} N:{N} C:{C}')
        qkv = (
            self.qkv(x1)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        ) 
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].
        # print(f'shape q: {q.shape}')
        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].
        # print(f'attention: {factor_att.shape}')
        # Convolutional relative position encoding.
        # size=(height, width)
        # crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].
        # Merge and reshape.
        # out = self.scale * factor_att + crpe
        out = self.scale * factor_att 
        out = (
            out.transpose(1, 2).reshape(B, N, C).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        out = self.proj(out) #[B, N, C]
        out = out.permute(0,2,1).reshape(B,C,numpath,height, width)#[B,C,N]
        # print(f'out shape: {out.shape}')
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out


class SE_Block(nn.Module):
    def __init__(self, in_ch, out_ch, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_ch, in_ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch, bias=False),
            nn.Sigmoid()
        )
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size = (1,1))
        self.act  = torch.nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # print('start using se block')
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.conv(x)
        x = self.act(self.bn(x))
        
        return x


class Conv3d_BN_channel_attention_concat(nn.Module):
    # input: attention list of length #path
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act_layer=None,
        norm_cfg="BN",
        cam='cam'
    ):
        super().__init__()
    
        self.bn = nn.BatchNorm2d(out_ch)

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
       
        self.interact_concat = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(4,1,1)),
            nn.GELU()
        )
        if cam == 'cam':
            self.channelAttention = CAM_Module(in_dim = in_ch)
            # print('cam chosen')
        elif cam == 'cam_fact':
            self.channelAttention = CAM_Factorized_Module(in_dim = in_ch)
            # print('factorized module chosen')
        else:
            self.channelAttention = CAM_Factorized_Module(in_dim = in_ch)
   

        self.bn3d = nn.BatchNorm3d(in_ch)

    def forward(self, x_list):
        b,c,h,w = x_list[0].shape
        # print(f"b:{b} c:{c} h:{h} w:{w}")
     
        out_3d = []
        for ip in range(len(x_list)):
#             print(f'ip:{ip}')
            in_x = x_list[ip]  
#             print(f'in each {ip} iteration: {in_x.shape}')
            in_x = in_x.unsqueeze_(dim=2)
            out_3d.append(in_x)
#             print(f'{ip} path extend to shape: {in_x.shape}')

            x = torch.cat(out_3d, dim=2)
            x = self.bn3d(x)
        

        # print(f'before channel attention: {x.shape}')
        x = self.channelAttention(x)
        x = self.bn3d(x)
        # print(f'after channel: {x.shape}')
        x = torch.squeeze(self.interact_concat(x), dim=2)
        # print(f'after squeeze: {x.shape}')
        x = self.bn(x)

        return x



class DWCPatchEmbed(nn.Module):
    """
    Depthwise Convolutional Patch Embedding layer
    Image to Patch Embedding
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=768,
        patch_size=16,
        stride=1,
        pad=0,
        act_layer=nn.Hardswish,
        norm_cfg='BN',
    ):
        super().__init__()
        self.stride = stride
        # TODO : confirm whether act_layer is effective or not
        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        x = self.patch_conv(x)
        # print(f'stride: {self.stride}')

        return x

class Patch_Embed_stage(nn.Module):

    def __init__(self, embed_dim, num_path=3, isPool=False, norm_cfg=dict(type="BN")):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList(
            [
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=2 if isPool and idx == 0 else 1,
                    pad=1,
                    norm_cfg='BN',
                )
                for idx in range(num_path)
            ]
        )

        # scale

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            # print(f'patch_embedding:{x.shape}')
            att_inputs.append(x)

        return att_inputs

class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous() #B C N->B N C

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
                )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = (
            x.transpose(1, 2).reshape(B, N, C).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


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


class MHCABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3,
        drop_path=0.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer= 'LN',
        shared_cpe=None,
        shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = MixFFN_skip(dim, dim * mlp_ratio)
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = nn.LayerNorm(dim,eps=1e-6)
        self.norm2 = nn.LayerNorm(dim,eps=1e-6)

    def forward(self, x, size):
        # x.shape = [B, N, C]
        # print('inside the MHCABlock')
        H,W = size
        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.factoratt_crpe(cur, size)

        cur = self.norm2(x)
        x = x + self.mlp(cur,H,W)
        return x


class MHCAEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        drop_path_list=[],
        qk_scale=None,
        crpe_window={3: 2, 5: 3, 7: 3},
    ):
        super().__init__()

        self.num_layers = num_layers
       
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window=crpe_window)
        self.MHCA_layers = nn.ModuleList(
            [
                MHCABlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_list[idx],
                    qk_scale=qk_scale,
                    shared_cpe=self.cpe,
                    shared_crpe=self.crpe,
                )
                for idx in range(self.num_layers)
            ]
        )

    def forward(self, x, size):
       
        H, W = size
        # print('inside the MHCAEncoder')
        B = x.shape[0]
        # x' shape : [B, N, C]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))
          

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Hardswish,
        norm_cfg="BN",
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(
            in_features, hidden_features, act_layer=act_layer, norm_cfg=norm_cfg
        )
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        # self.norm = norm_layer(hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, torch.sqrt(2.0 / fan_out))
            # if m.bias is not None:
            #     m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


    def forward(self, x):
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat



class SK_Block(nn.Module):
    #input: list of len=num_path b c h w

    def __init__(self, in_ch, out_ch, num_path=3,reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,in_ch//reduction)

        self.fc=nn.Linear(in_ch,self.d)
        
        self.fcs=nn.ModuleList([])
        for i in range(num_path):
            self.fcs.append(nn.Linear(self.d,in_ch))
        self.softmax=nn.Softmax(dim=0)
        self.conv_bn_ac = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = (1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
            
        )



    def forward(self, x):
        # print("checking skn")
        # print(f"x[0].shape: {x[0].shape}")
        bs, c, h, w = x[0].size()
    
        feats=torch.stack(x,0)#k,bs,channel,h,w
        # print(f'feats.shape: {feats.shape}')

        ### fuse
        U=sum(x) #bs,c,h,w
        # print(f'U.shape: {U.shape}')

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d
        # print(f'Z.shape: {Z.shape}')

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weights=torch.stack(weights,0)#k,bs,channel,1,1
        # print(f'attention_weight: {attention_weights.shape}')
        attention_weights=self.softmax(attention_weights)#k,bs,channel,1,1

        ### fuse
        V=(attention_weights*feats).sum(0)
        out = self.conv_bn_ac(V)
        # print(f'out: {out.shape}')
        
        return out

        


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """
    Generate drop path rate list following linear decay rule
    """
    dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur : cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr

#cbam

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,stride=1, padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
#         print(f'shape of max_result: {max_result.shape}')
        avg_result=torch.mean(x,dim=1,keepdim=True)
#         print(f'shape of avg_result: {avg_result.shape}')
        result=torch.cat([max_result,avg_result],1)
#         print(f'shape of result: {result.shape}')
        output=self.conv(result)
#         print(f'shape after conv: {output.shape}')
        output=self.sigmoid(output)

        return output



class CBAMBlock(nn.Module):
    
    def __init__(self, channel=512,out_ch=512, use_sa=True, reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        self.conv2d_bn_act = nn.Sequential(
            nn.Conv2d(channel,out_ch,1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
            
        )
        self.use_sa = use_sa


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # print('checking cbam')
        b, c, _, _ = x.size()
        residual=x
#         print(f'shape of residual: {x.shape}')
        out=x*self.ca(x)
        # print(f'shape of out ca: {out.shape}')
        if self.use_sa:
            out=out*self.sa(out)
            # print(f'shape of out sa: {out.shape}')
        out_cat = out+residual
        out_cat = self.conv2d_bn_act(out_cat)
        
        return out_cat

class CBAMBlock_casa(nn.Module):

    def __init__(self, channel=512,out_ch=512, use_sa=True, reduction=16,kernel_size=49, inter="res"):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        self.conv2d_bn_act = nn.Sequential(
            nn.Conv2d(channel,out_ch,1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
            
        )
        self.use_sa = use_sa
        self.inter = inter


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # print('checking cbam')
        # print(f"cama: len of x_input: {len(x)}")
        residual = x[0]
        # out_tr = torch.cat(x[1:], dim=1) #concat the outputs of transformer
        out_cat = torch.cat(x, dim=1) 
        b, c, _, _ = out_cat.size()
     
        # print(f'shape of residual: {residual.shape}')
        # print(f'shape of out_tr: {out_cat.shape}')
        out=out_cat*self.ca(out_cat)
        # print(f'shape of out ca: {out.shape}')
        if self.use_sa and self.inter == 'res':
            out=out*self.sa(residual)
            print("use res inter")
        elif self.use_sa and self.inter == 'out':
            out=out*self.sa(out)# I try to replace it with residual or out
            print("use out inter")
        else:
            out=out
            # print(f'shape of out sa: {out.shape}')
        out_cat = out+out_cat
        out_cat = self.conv2d_bn_act(out_cat)
        
        return out_cat

## CoordAttention
class silu_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(silu_sigmoid, self).__init__()
        self.silu = nn.SiLU(inplace=inplace)
    def forward(self,x):
        x = self.silu(x+3)/6
        upper = torch.ones(x.shape).cuda()
        return torch.minimum(x,upper)

class silu_swish(nn.Module):
    def __init__(self, inplace=True):
        super(silu_swish, self).__init__()
        self.sigmoid = silu_sigmoid(inplace=inplace)

    def forward(self, x):
        # print('silu swish')
        return x * self.sigmoid(x)

# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace) #min(max(0,x),6)

#     def forward(self, x):
#         return self.relu(x + 3) / 6

# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)

#     def forward(self, x):
#         return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = silu_swish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        
        self.conv_in_out = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        # print("Using coord")
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
#         print(f"x_h:{x_h.shape}")
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
#         print(f"x_w:{x_w.shape}")

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)

        y = self.bn1(y)
        y = self.act(y) 

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)


        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        out = self.conv_in_out(out)

        return out

class MHCA_stage(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_embed_dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        num_path=4,
        norm_cfg="BN",
        drop_path_list=[],
        concat='normal',
        use_sa = True,
        sa_ker = 7
    ):
        super().__init__()
        self.concat=concat
      
        self.mhca_blks = nn.ModuleList(
            [
                MHCAEncoder(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                )
                for _ in range(num_path)
            ]
        )

        self.InvRes = ResBlock(
            in_features=embed_dim, out_features=embed_dim, norm_cfg=norm_cfg
        )
        if self.concat == 'normal':
            self.aggregate = Conv2d_BN(
                embed_dim * (num_path + 1),
                out_embed_dim,
                act_layer=nn.Hardswish,
                norm_cfg=norm_cfg,
            )
        elif self.concat == '3d':
            self.aggregate = Conv3d_BN_concat(
                embed_dim,
                out_embed_dim
            )
        elif self.concat == 'se':
            self.aggregate = SE_Block(in_ch = embed_dim*(num_path+1), out_ch= out_embed_dim, r=16)
        elif self.concat == 'skn':
            self.aggregate = SK_Block(in_ch=embed_dim, out_ch=out_embed_dim, num_path=num_path+1,reduction=8)
        elif self.concat == 'cbam':
            self.aggregate = CBAMBlock(channel=embed_dim*(num_path+1),out_ch=out_embed_dim,use_sa=use_sa, reduction=16,kernel_size=sa_ker)
        elif self.concat == 'coord':
            self.aggregate = CoordAtt(inp=embed_dim*(num_path+1),oup=out_embed_dim,reduction=16)
        else:
            self.aggregate = Conv3d_BN_channel_attention_concat(
                embed_dim,
                out_embed_dim,
                cam=self.concat# cam or cam_fact
            )


    def forward(self, inputs):
        # print(len(inputs))
        # print("---Res---")
        att_outputs = [self.InvRes(inputs[0])]
       
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            # print(f'H:{H} W:{W}')
            x = x.flatten(2).transpose(1, 2).contiguous()
            # print('going to the encoder')
            tmp = encoder(x, size=(H, W))
            # print(f'\n---attention path{tmp.shape}--')
            att_outputs.append(tmp)

        # out_concat = torch.cat(att_outputs, dim=1)
        # print(f'att_outputs: {len(att_outputs)}')
        # print(f'before cat: {att_outputs[0].shape}')
        # print(f'before cat: {att_outputs[2].shape}')
        # print(self.concat)
        if self.concat == 'normal' or self.concat == 'se' or self.concat == 'cbam' or self.concat == "coord":
            out = self.aggregate( torch.cat(att_outputs, dim=1))
            # print('this one?')
        else:
            # print('sent a listinto')
            out = self.aggregate(att_outputs)

        # print(f'\n after cat: {out.shape}')

        return out

class MHCA_stage_casa(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_embed_dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        num_path=4,
        norm_cfg="BN",
        drop_path_list=[],
        concat='normal',
        use_sa = True,
        sa_ker = 7,
        inter = 'res'
    ):
        super().__init__()
        self.concat=concat
      
        self.mhca_blks = nn.ModuleList(
            [
                MHCAEncoder(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                )
                for _ in range(num_path)
            ]
        )

        self.InvRes = ResBlock(
            in_features=embed_dim, out_features=embed_dim, norm_cfg=norm_cfg
        )
        if self.concat == 'normal':
            self.aggregate = Conv2d_BN(
                embed_dim * (num_path + 1),
                out_embed_dim,
                act_layer=nn.Hardswish,
                norm_cfg=norm_cfg,
            )
        elif self.concat == '3d':
            self.aggregate = Conv3d_BN_concat(
                embed_dim,
                out_embed_dim
            )
        elif self.concat == 'se':
            self.aggregate = SE_Block(in_ch = embed_dim*(num_path+1), out_ch= out_embed_dim, r=16)
        elif self.concat == 'skn':
            self.aggregate = SK_Block(in_ch=embed_dim, out_ch=out_embed_dim, num_path=num_path+1,reduction=8)
        elif self.concat == 'cbam':
            self.aggregate = CBAMBlock_casa(channel=embed_dim*(num_path+1),out_ch=out_embed_dim,use_sa=use_sa, reduction=16,kernel_size=sa_ker, inter=inter)
        else:
            self.aggregate = Conv3d_BN_channel_attention_concat(
                embed_dim,
                out_embed_dim,
                cam=self.concat# cam or cam_fact
            )


    def forward(self, inputs):
        # print(len(inputs))
        # print("---Res---")
        att_outputs = [self.InvRes(inputs[0])]
       
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            # print(f'H:{H} W:{W}')
            x = x.flatten(2).transpose(1, 2).contiguous()
            # print('going to the encoder')
            tmp = encoder(x, size=(H, W))
            # print(f'\n---attention path{tmp.shape}--')
            att_outputs.append(tmp)

        # out_concat = torch.cat(att_outputs, dim=1)
        # print(f'att_outputs: {len(att_outputs)}')
        # print(f'before cat: {att_outputs[0].shape}')
        # print(f'before cat: {att_outputs[2].shape}')
        # print(self.concat)
        if self.concat == 'normal' or self.concat == 'se':
            out = self.aggregate( torch.cat(att_outputs, dim=1))
            # print('this one?')
        else:
            # print('sent a listinto')
            out = self.aggregate(att_outputs)

        # print(f'\n after cat: {out.shape}')

        return out


class MSViT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip',MSViT_config=1, concat='normal', use_sa_list=[True, True, False],
        sa_ker=7):
        super().__init__()

        self.Hs=[56, 28, 14, 7]
        self.Ws=[56, 28, 14, 7]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        # dil_conv = False #no dilation this version...
        if dil_conv:  
            dilation = 2 
            patch_sizes1 = [7, 5, 5, 5]
            patch_sizes2 = [0, 3, 3, 3]
            patch_sizes3 = [0, 1, 1, 1]
            dil_padding_sizes1 = [3, 0, 0, 0]    
            dil_padding_sizes2 = [0, 0, 0, 0]
            dil_padding_sizes3 = [0, 0, 0, 0]
            
        else:
            dilation = 1
            patch_sizes1 = [7, 3, 3, 3]
            patch_sizes2 = [5, 1, 1, 1]
            patch_sizes3 = [0, 5, 5, 5]
            dil_padding_sizes1 = [3, 1, 1, 1]
            dil_padding_sizes2 = [1, 0, 0, 0]
            dil_padding_sizes3 = [1, 2, 2, 2]


        # 1 by 1 convolution to alter the dimension
        self.conv1_1_s1 = nn.Conv2d(3*in_dim[0], in_dim[0], 1)
        self.conv1_1_s2 = nn.Conv2d(3*in_dim[1], in_dim[1], 1)
        self.conv1_1_s3 = nn.Conv2d(3*in_dim[2], in_dim[2], 1)
        self.conv1_1_s4 = nn.Conv2d(3*in_dim[3], in_dim[3], 1)

        # -------------MSViT codes---------------------------------------------------
        # Patch embeddings.
        if MSViT_config == 1:
            num_path = [3,3,3]
            # num_layers = [3,6,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]
        elif MSViT_config == 2:
            num_path = [3,3,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]
        else: #config==2
            num_path = [3,3,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]



        mlp_ratios = [4,4,4] # what is mlp_ratios?
        num_stages = 3
        drop_path_rate=0.0
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)
     
        self.patch_embed_stage2 = Patch_Embed_stage(
                    in_dim[0],
                    num_path=num_path[0],
                    isPool=True,
                    norm_cfg='BN',
                )

        self.patch_embed_stage3 = Patch_Embed_stage(
                    in_dim[1],
                    num_path=num_path[1],
                    isPool=True, 
                    norm_cfg='BN',
                )
        # if idx == 0 else True
        self.patch_embed_stage4 = Patch_Embed_stage(
                    in_dim[2],
                    num_path=num_path[2],
                    isPool=True,
                    norm_cfg='BN',
                )


        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stage2 = MHCA_stage(
                    in_dim[0],
                    in_dim[1],
                    num_layers[0],
                    num_heads[0],
                    mlp_ratios[0],
                    num_path[0],
                    norm_cfg='BN',
                    drop_path_list=dpr[0],
                    concat=concat,
                    use_sa = use_sa_list[0],
                    sa_ker=sa_ker

                )

        self.mhca_stage3 = MHCA_stage(
                    in_dim[1],
                    in_dim[2],
                    num_layers[1],
                    num_heads[1],
                    mlp_ratios[1],
                    num_path[1],
                    norm_cfg='BN',
                    drop_path_list=dpr[1],
                    concat=concat,
                    use_sa = use_sa_list[1],
                    sa_ker=sa_ker
                )

        self.mhca_stage4 = MHCA_stage(
                    in_dim[2],
                    in_dim[3],
                    num_layers[2],
                    num_heads[2],
                    mlp_ratios[2],
                    num_path[2],
                    norm_cfg='BN',
                    drop_path_list=dpr[2],
                    concat=concat,
                    use_sa = use_sa_list[2],
                    sa_ker=sa_ker
                )

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0])
        
       
        # # transformer encoder
        self.cpe = ConvPosEnc(in_dim[0], k=3)
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        
        
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # trunc_normal_(m.weight, std=0.02)
                # if isinstance(m, nn.Linear) and m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                # nn.init.constant_(m.bias, 0)
                # nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

       
            
        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []
        # stage conv stem
        # stage 1
        x, H, W = self.patch_embed1(x)
        # print(f'stage1 after embedding: {x.shape}')
        # x = self.cpe(x, (H,W))
        # print(f'stage1 after cpe: {x.shape}')
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

      

        # # stage 2
        # print("-------EN: Stage 2------\n\n")
        att_input = self.patch_embed_stage2(x)
        x = self.mhca_stage2(att_input)
        outs.append(x)

        # # stage 3
        # print("-------EN: Stage 3------\n\n")
        att_input = self.patch_embed_stage3(x)
        x = self.mhca_stage3(att_input)
        outs.append(x)

        # # stage 4
        # print("-------EN: Stage 4------\n\n")
        att_input = self.patch_embed_stage4(x)
        x = self.mhca_stage4(att_input)
        outs.append(x)

        return outs
    
class MSViT_4Stages(nn.Module):
        
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip',MSViT_config=1, concat='normal', use_sa_list=[True,True,True,False], sa_ker=7):
        super().__init__()

        self.Hs=[56, 28, 14, 7]
        self.Ws=[56, 28, 14, 7]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        # dil_conv = False #no dilation this version...
        if dil_conv:  
            dilation = 2 
            patch_sizes1 = [7, 5, 5, 5]
            patch_sizes2 = [0, 3, 3, 3]
            patch_sizes3 = [0, 1, 1, 1]
            dil_padding_sizes1 = [3, 0, 0, 0]    
            dil_padding_sizes2 = [0, 0, 0, 0]
            dil_padding_sizes3 = [0, 0, 0, 0]
            
        else:
            dilation = 1
            patch_sizes1 = [7, 3, 3, 3]
            patch_sizes2 = [5, 1, 1, 1]
            patch_sizes3 = [0, 5, 5, 5]
            dil_padding_sizes1 = [3, 1, 1, 1]
            dil_padding_sizes2 = [1, 0, 0, 0]
            dil_padding_sizes3 = [1, 2, 2, 2]


        # 1 by 1 convolution to alter the dimension
        self.conv1_1_s1 = nn.Conv2d(3*in_dim[0], in_dim[0], 1)
        self.conv1_1_s2 = nn.Conv2d(3*in_dim[1], in_dim[1], 1)
        self.conv1_1_s3 = nn.Conv2d(3*in_dim[2], in_dim[2], 1)
        self.conv1_1_s4 = nn.Conv2d(3*in_dim[3], in_dim[3], 1)

        # -------------MSViT codes---------------------------------------------------
        
        # Patch embeddings.
        if MSViT_config == 1:
            num_path = [2,3,3,3]
            num_layers = [1, 3,8,3]
            num_heads = [8,8,8,8]
        elif MSViT_config == 2:
            num_path = [2,3,3,3]
            num_layers = [1, 3,8,3]
            num_heads = [8,8,8,8]
        else: #config==2
            num_path = [2,3,3,3]
            num_layers = [1, 3,8,3]
            num_heads = [8,8,8,8]


    
        mlp_ratios = [4, 4, 4, 4] # what is mlp_ratios?
        num_stages = 4
        drop_path_rate=0.0
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)
        # print(f'dpr: {dpr[2]}')

        self.stem = nn.Sequential(
            Conv2d_BN(
                3,
                in_dim[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                in_dim[0] // 2,
                in_dim[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        self.patch_embed_stage1 = Patch_Embed_stage(
                    in_dim[0],
                    num_path=num_path[0],
                    isPool=False,
                    norm_cfg='BN',
                )
     
        self.patch_embed_stage2 = Patch_Embed_stage(
                    in_dim[0],
                    num_path=num_path[1],
                    isPool=True,
                    norm_cfg='BN',
                )

        self.patch_embed_stage3 = Patch_Embed_stage(
                    in_dim[1],
                    num_path=num_path[2],
                    isPool=True, 
                    norm_cfg='BN',
                )
        # if idx == 0 else True
        self.patch_embed_stage4 = Patch_Embed_stage(
                    in_dim[2],
                    num_path=num_path[3],
                    isPool=True,
                    norm_cfg='BN',
                )


        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stage1 = MHCA_stage(
                    in_dim[0],
                    in_dim[0],
                    num_layers[0],
                    num_heads[0],
                    mlp_ratios[0],
                    num_path[0],
                    norm_cfg='BN',
                    drop_path_list=dpr[0],
                    concat=concat,
                    use_sa = use_sa_list[0],
                    sa_ker=sa_ker
                )

        self.mhca_stage2 = MHCA_stage(
                    in_dim[0],
                    in_dim[1],
                    num_layers[1],
                    num_heads[1],
                    mlp_ratios[1],
                    num_path[1],
                    norm_cfg='BN',
                    drop_path_list=dpr[1],
                    concat=concat,
                    use_sa = use_sa_list[1],
                    sa_ker=sa_ker
                )

        self.mhca_stage3 = MHCA_stage(
                    in_dim[1],
                    in_dim[2],
                    num_layers[2],
                    num_heads[2],
                    mlp_ratios[2],
                    num_path[2],
                    norm_cfg='BN',
                    drop_path_list=dpr[2],
                    concat=concat,
                    use_sa = use_sa_list[2],
                    sa_ker=sa_ker
                )

        self.mhca_stage4 = MHCA_stage(
                    in_dim[2],
                    in_dim[3],
                    num_layers[3],
                    num_heads[3],
                    mlp_ratios[3],
                    num_path[3],
                    norm_cfg='BN',
                    drop_path_list=dpr[3],
                    concat=concat,
                    use_sa = use_sa_list[3],
                    sa_ker=sa_ker
                )
    
        # # transformer encoder
        self.cpe = ConvPosEnc(in_dim[0], k=3)
        # self.block1 = nn.ModuleList([ 
        #     EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        # for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        
        
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # trunc_normal_(m.weight, std=0.02)
                # if isinstance(m, nn.Linear) and m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                # nn.init.constant_(m.bias, 0)
                # nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

       
            
        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("The 4 stages:")
        B = x.shape[0]
        outs = []
        # stage conv stem
        # stage 1
   
        x = self.stem(x)
      
        att_input = self.patch_embed_stage1(x)
 
        x = self.mhca_stage1(att_input)
        outs.append(x)
      

        # # stage 2
        # print("-------EN: Stage 2------\n\n")
        att_input = self.patch_embed_stage2(x)
  
        x = self.mhca_stage2(att_input)
        outs.append(x)

        # # stage 3
        # print("-------EN: Stage 3------\n\n")
        att_input = self.patch_embed_stage3(x)
        x = self.mhca_stage3(att_input)
        outs.append(x)

        # # stage 4
        # print("-------EN: Stage 4------\n\n")
        att_input = self.patch_embed_stage4(x)
        x = self.mhca_stage4(att_input)
        outs.append(x)

        return outs
  
class MSViT_casa(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip',MSViT_config=1, concat='normal', use_sa_list=[True, True, False],
        sa_ker=7, inter='res'):
        super().__init__()

        self.Hs=[56, 28, 14, 7]
        self.Ws=[56, 28, 14, 7]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        # dil_conv = False #no dilation this version...
        if dil_conv:  
            dilation = 2 
            patch_sizes1 = [7, 5, 5, 5]
            patch_sizes2 = [0, 3, 3, 3]
            patch_sizes3 = [0, 1, 1, 1]
            dil_padding_sizes1 = [3, 0, 0, 0]    
            dil_padding_sizes2 = [0, 0, 0, 0]
            dil_padding_sizes3 = [0, 0, 0, 0]
            
        else:
            dilation = 1
            patch_sizes1 = [7, 3, 3, 3]
            patch_sizes2 = [5, 1, 1, 1]
            patch_sizes3 = [0, 5, 5, 5]
            dil_padding_sizes1 = [3, 1, 1, 1]
            dil_padding_sizes2 = [1, 0, 0, 0]
            dil_padding_sizes3 = [1, 2, 2, 2]


        # 1 by 1 convolution to alter the dimension
        self.conv1_1_s1 = nn.Conv2d(3*in_dim[0], in_dim[0], 1)
        self.conv1_1_s2 = nn.Conv2d(3*in_dim[1], in_dim[1], 1)
        self.conv1_1_s3 = nn.Conv2d(3*in_dim[2], in_dim[2], 1)
        self.conv1_1_s4 = nn.Conv2d(3*in_dim[3], in_dim[3], 1)

        # -------------MSViT codes---------------------------------------------------
        # Patch embeddings.
        if MSViT_config == 1:
            num_path = [3,3,3]
            # num_layers = [3,6,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]
        elif MSViT_config == 2:
            num_path = [3,3,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]
        else: #config==2
            num_path = [3,3,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]



        mlp_ratios = [4,4,4] # what is mlp_ratios?
        num_stages = 3
        drop_path_rate=0.0
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)
     
        self.patch_embed_stage2 = Patch_Embed_stage(
                    in_dim[0],
                    num_path=num_path[0],
                    isPool=True,
                    norm_cfg='BN',
                )

        self.patch_embed_stage3 = Patch_Embed_stage(
                    in_dim[1],
                    num_path=num_path[1],
                    isPool=True, 
                    norm_cfg='BN',
                )
        # if idx == 0 else True
        self.patch_embed_stage4 = Patch_Embed_stage(
                    in_dim[2],
                    num_path=num_path[2],
                    isPool=True,
                    norm_cfg='BN',
                )


        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stage2 = MHCA_stage_casa(
                    in_dim[0],
                    in_dim[1],
                    num_layers[0],
                    num_heads[0],
                    mlp_ratios[0],
                    num_path[0],
                    norm_cfg='BN',
                    drop_path_list=dpr[0],
                    concat=concat,
                    use_sa = use_sa_list[0],
                    sa_ker=sa_ker,
                    inter = inter

                )

        self.mhca_stage3 = MHCA_stage_casa(
                    in_dim[1],
                    in_dim[2],
                    num_layers[1],
                    num_heads[1],
                    mlp_ratios[1],
                    num_path[1],
                    norm_cfg='BN',
                    drop_path_list=dpr[1],
                    concat=concat,
                    use_sa = use_sa_list[1],
                    sa_ker=sa_ker,
                    inter = inter
                )

        self.mhca_stage4 = MHCA_stage_casa(
                    in_dim[2],
                    in_dim[3],
                    num_layers[2],
                    num_heads[2],
                    mlp_ratios[2],
                    num_path[2],
                    norm_cfg='BN',
                    drop_path_list=dpr[2],
                    concat=concat,
                    use_sa = use_sa_list[2],
                    sa_ker=sa_ker,
                    inter = inter
                )

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0])
        
       
        # # transformer encoder
        self.cpe = ConvPosEnc(in_dim[0], k=3)
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        
        
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # trunc_normal_(m.weight, std=0.02)
                # if isinstance(m, nn.Linear) and m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                # nn.init.constant_(m.bias, 0)
                # nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

       
            
        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []
        # stage conv stem
        # stage 1
        x, H, W = self.patch_embed1(x)
        # print(f'stage1 after embedding: {x.shape}')
        # x = self.cpe(x, (H,W))
        # print(f'stage1 after cpe: {x.shape}')
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

      

        # # stage 2
        # print("-------EN: Stage 2------\n\n")
        att_input = self.patch_embed_stage2(x)
        # print(len(att_input))
        x = self.mhca_stage2(att_input)
        outs.append(x)

        # # stage 3
        # print("-------EN: Stage 3------\n\n")
        att_input = self.patch_embed_stage3(x)
        # print(len(att_input))
        x = self.mhca_stage3(att_input)
        outs.append(x)

        # # stage 4
        # print("-------EN: Stage 4------\n\n")
        att_input = self.patch_embed_stage4(x)
        # print(len(att_input))
        x = self.mhca_stage4(att_input)
        outs.append(x)

        return outs
  
#------------------------------
#-------- Bridge -------------
#------------------------------
class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if(len(self.reduction_ratio)==4):
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim*2, dim*2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim*5, dim*5, reduction_ratio[1], reduction_ratio[1])
        
        elif(len(self.reduction_ratio)==3):
            self.sr0 = nn.Conv2d(dim*2, dim*2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim*5, dim*5, reduction_ratio[1], reduction_ratio[1])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if(len(self.reduction_ratio)==4):
            tem0 = x[:,:3136,:].reshape(B, 56, 56, C).permute(0, 3, 1, 2) 
            tem1 = x[:,3136:4704,:].reshape(B, 28, 28, C*2).permute(0, 3, 1, 2)
            tem2 = x[:,4704:5684,:].reshape(B, 14, 14, C*5).permute(0, 3, 1, 2)
            tem3 = x[:,5684:6076,:]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))
        
        if(len(self.reduction_ratio)==3):
            tem0 = x[:,:1568,:].reshape(B, 28, 28, C*2).permute(0, 3, 1, 2) 
            tem1 = x[:,1568:2548,:].reshape(B, 14, 14, C*5).permute(0, 3, 1, 2)
            tem2 = x[:,2548:2940,:]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            
            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))
        
        return reduce_out

        


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim,reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"\n input of EfficientSelfAtten:{x.shape}")
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        # print(f"Shape of q: {q.shape}")

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)
            # print(f"Shape of reduced x for k and v: {x.shape}")

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print(f"Shape of k:{k.shape}; shape of v {v.shape}")

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print(f"Shape of attention matrix:{attn.shape}")
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2)
        # print(f"Shape of attention result:{x_atten.shape}")
        x_atten = x_atten.reshape(B, N, C)
        out = self.proj(x_atten)
        # print(f'out of attention: {out.shape}')


        return out


class M_EfficientChannelAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim,reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"\n inside channel att")
        B, N, C = x.shape
        k = self.k(x).reshape((B, C, N))
        q = self.q(x).reshape((B, C, N))
        v = self.v(x).reshape((B, C, N))

        # q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        head_k_ch = C // self.head
        head_v_ch = C// self.head
        
        attended_values = []
        for i in range(self.head):
            key = F.softmax(k[
                :,
                i * head_k_ch: (i + 1) * head_k_ch,
                :
            ], dim=2)
            
            query = F.softmax(q[
                :,
                i * head_k_ch: (i + 1) * head_k_ch,
                :
            ], dim=1)
                        
            value = v[
                :,
                i * head_v_ch: (i + 1) * head_v_ch,
                :
            ]            
            
            context = key @ value.transpose(1, 2) # dk*dv
            # print(f'context:{context.shape}')  
            attended_value = (context.transpose(1, 2) @ query)
            # print(f'attended_value:{attended_value.shape}')    
            attended_value = attended_value.reshape(B, head_v_ch, N) # n*dv      
            # print(f'reshaped attended_value:{attended_value.shape}')      
            attended_values.append(attended_value)
                
        aggregated_values = torch.cat(attended_values, dim=1)
        # print(f'aggregated_values: {aggregated_values.shape}')
        out = self.proj(aggregated_values.permute((0,2,1)))
        # print(f'out of attention: {out.shape}')

        return out


class BridgLayer_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios, ch_att):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        if ch_att:
            self.attn = M_EfficientChannelAtten(dims, head, reduction_ratios)
        else:
            self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims,dims*4)
        self.mixffn2 = MixFFN_skip(dims*2,dims*8)
        self.mixffn3 = MixFFN_skip(dims*5,dims*20)
        self.mixffn4 = MixFFN_skip(dims*8,dims*32)
        
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)


        tem1 = tx[:,:3136,:].reshape(B, -1, C) 
        tem2 = tx[:,3136:4704,:].reshape(B, -1, C*2)
        tem3 = tx[:,4704:5684,:].reshape(B, -1, C*5)
        tem4 = tx[:,5684:6076,:].reshape(B, -1, C*8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        
        tx2 = tx1 + t1


        return tx2



class BridgeBlock_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios, br_ch_att_list):
        super().__init__()
        
        self.bridge_layer1 = BridgLayer_4(dims, head, reduction_ratios, br_ch_att_list[0])
        self.bridge_layer2 = BridgLayer_4(dims, head, reduction_ratios, br_ch_att_list[1])
        self.bridge_layer3 = BridgLayer_4(dims, head, reduction_ratios, br_ch_att_list[2])
        self.bridge_layer4 = BridgLayer_4(dims, head, reduction_ratios, br_ch_att_list[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('Checking bridge')
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B,_,C = bridge4.shape
        outs = []

        sk1 = bridge4[:,:3136,:].reshape(B, 56, 56, C).permute(0,3,1,2) 
        sk2 = bridge4[:,3136:4704,:].reshape(B, 28, 28, C*2).permute(0,3,1,2) 
        sk3 = bridge4[:,4704:5684,:].reshape(B, 14, 14, C*5).permute(0,3,1,2) 
        sk4 = bridge4[:,5684:6076,:].reshape(B, 7, 7, C*8).permute(0,3,1,2) 

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs

class BridgLayer_para(nn.Module):
    def __init__(self, dims, head, reduction_ratios, ch_att):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        if ch_att:
            self.attn = M_EfficientChannelAtten(dims, head, reduction_ratios)
        else:
            self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims,dims*4)
        self.mixffn2 = MixFFN_skip(dims*2,dims*8)
        self.mixffn3 = MixFFN_skip(dims*5,dims*20)
        self.mixffn4 = MixFFN_skip(dims*8,dims*32)
        
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)


        tem1 = tx[:,:3136,:].reshape(B, -1, C) 
        tem2 = tx[:,3136:4704,:].reshape(B, -1, C*2)
        tem3 = tx[:,4704:5684,:].reshape(B, -1, C*5)
        tem4 = tx[:,5684:6076,:].reshape(B, -1, C*8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        
        tx2 = tx1 + t1


        return tx2


class BridgeBlock_para(nn.Module):
    def __init__(self, dims, head, reduction_ratios, br_ch_att_list):
        super().__init__()
        
        self.bridge_layer1 = BridgLayer_para(dims, head, reduction_ratios, True)#channel
        self.bridge_layer2 = BridgLayer_para(dims, head, reduction_ratios, False)#spatial
        self.proj_act = nn.Sequential(
            nn.Linear(2*dims, dims),
            nn.LayerNorm(dims),
            nn.GELU()
            )
        self.bridge_layer3 = BridgLayer_para(dims, head, reduction_ratios, False)
        self.bridge_layer4 = BridgLayer_para(dims, head, reduction_ratios, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('Checking bridge')
        bridge1 = self.bridge_layer1(x)#[B N C]
        bridge2 = self.bridge_layer2(x)#[B N C]
        br_dual = torch.cat((bridge1,bridge2), dim=2)#[B N 2C]
        # print(f"shape of concat br_dual:{br_dual.shape}")
        br_dual = self.proj_act(br_dual)
        # print(f"shape of processed br_dual:{br_dual.shape}")
        bridge3 = self.bridge_layer3(br_dual)
        bridge4 = self.bridge_layer4(bridge3)

        B,_,C = bridge4.shape
        outs = []

        sk1 = bridge4[:,:3136,:].reshape(B, 56, 56, C).permute(0,3,1,2) 
        sk2 = bridge4[:,3136:4704,:].reshape(B, 28, 28, C*2).permute(0,3,1,2) 
        sk3 = bridge4[:,4704:5684,:].reshape(B, 14, 14, C*5).permute(0,3,1,2) 
        sk4 = bridge4[:,5684:6076,:].reshape(B, 7, 7, C*8).permute(0,3,1,2) 

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs

# ----New Bridge---------
# Spatial Fuse module
class MultiScaleAtten(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head)**0.5

    def forward(self, x):
        B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
        qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous() # (3, B, num_block, num_block, head, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
        atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
        return atten_value


class InterTransBlock(nn.Module):
    def __init__(self, dim):
        super(InterTransBlock, self).__init__()
        self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = MultiScaleAtten(dim)
        self.mlp = MLP_FFN(dim,4*dim)

    def forward(self, x):
        h = x  # (B, N, H)
        x = self.SlayerNorm_1(x)

        x = self.Attention(x)  
        x = h + x

        h = x
        x = self.SlayerNorm_2(x)

        x = self.mlp(x)
        x = h + x

        return x


class SpatialAwareTrans(nn.Module):
    def __init__(self, dim=64, num_sp_layer=1):  # (224*64, 112*128, 56*256, 28*256, 14*512) dim = 256
        super(SpatialAwareTrans, self).__init__()
        self.win_size_list = [8,4,2,1]
        self.channels = [64, 64*2, 64*5, 64*8]
        self.dim = dim
        self.depth = 4
        self.fc1 = nn.Linear(self.channels[0],dim)
        self.fc2 = nn.Linear(self.channels[1],dim)
        self.fc3 = nn.Linear(self.channels[2],dim)
        self.fc4 = nn.Linear(self.channels[3],dim)

        self.fc1_back = nn.Linear(dim, self.channels[0])
        self.fc2_back = nn.Linear(dim, self.channels[1])
        self.fc3_back = nn.Linear(dim, self.channels[2])
        self.fc4_back = nn.Linear(dim, self.channels[3])

        self.fc_back = nn.ModuleList()
        for i in range(self.depth):
            self.fc_back.append(nn.Linear(self.dim, self.channels[i]))
      
        self.num = num_sp_layer # the number of layers
    

        self.group_attention = []
        for i in range(self.num):
            self.group_attention.append(InterTransBlock(dim))
        self.group_attention = nn.Sequential(*self.group_attention)
        self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]

    def forward(self, x):
        # project channel dimension to 256
        # print("Start spatial aware:------------")
        # print(f"x_0:{x[0].shape}")
        # print(f"x_1:{x[1].shape}")
        # print(f"x_2:{x[2].shape}")
        # print(f"x_3:{x[3].shape}")

        # utilize linear to project from other channel number to 256(C)
        x[0] = self.fc1(x[0].permute(0,2,3,1))
        x[1] = self.fc2(x[1].permute(0,2,3,1))
        x[2] = self.fc3(x[2].permute(0,2,3,1))
        x[3] = self.fc4(x[3].permute(0,2,3,1))
        # x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
        # Patch Matching
        # print("-----------------")
        for j, item in enumerate(x):
            # print(f"#{j} shape: {item.shape}")
            B, H, W, C = item.shape
            win_size = self.win_size_list[j]
            # print(f'window size: {win_size}')
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous()#([B,H/win,W/win, win,win,C])
            # print(f'reshape first step:{item.shape}')
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()#([B,H/win,W/win, win*win,C])
            # print(f'reshape second step:{item.shape}')
            x[j] = item
        x = tuple(x)
        x = torch.cat(x, dim=-2)  # (B, H // win, W // win, N, C)
        # print(f"\n fuse the four level together:{x.shape}")
        
        # Scale fusion
        for i in range(self.num):
            # print("num of scale fusion")
            x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)

        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)
        # patch reversion
        # print("-------reversion----------")
        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.win_size_list[j]
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, num_blocks*win_size, num_blocks*win_size, C)
            item = self.fc_back[j](item).permute(0, 3, 1, 2).contiguous()
            # print(f"#{j} shape: {item.shape}")
            x[j] = item
       
        return x



    
class BridgeLayer_new(nn.Module):
    def __init__(self, dims, head, reduction_ratios, num_sp):
        super().__init__()
        C = 64
        
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims,dims*4)
        self.mixffn2 = MixFFN_skip(dims*2,dims*8)
        self.mixffn3 = MixFFN_skip(dims*5,dims*20)
        self.mixffn4 = MixFFN_skip(dims*8,dims*32)
        
        self.num = num_sp
        self.scale_fuse_att = SpatialAwareTrans(dim=dims, num_sp_layer=num_sp)
        
    def forward(self, inputs):
        # print(f"num_sp:{self.num}")
        B = inputs[0].shape[0]
        C = 64
        H = 56
        W = 56
        if (type(inputs) == list):
            if self.num > 0:
                inputs = self.scale_fuse_att(inputs)
            c1, c2, c3, c4 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)


        tem1 = tx[:,:3136,:].reshape(B, -1, C) 
        tem2 = tx[:,3136:4704,:].reshape(B, -1, C*2)
        tem3 = tx[:,4704:5684,:].reshape(B, -1, C*5)
        tem4 = tx[:,5684:6076,:].reshape(B, -1, C*8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        
        tx2 = tx1 + t1


        return tx2



class BridgeBlock_sp(nn.Module):
    def __init__(self, dims, head, reduction_ratios, num_sp):
        super().__init__()
    
        self.bridge_layer1 = BridgeLayer_new(dims, head, reduction_ratios, num_sp)
        self.bridge_layer2 = BridgeLayer_new(dims, head, reduction_ratios, num_sp)
        self.bridge_layer3 = BridgeLayer_new(dims, head, reduction_ratios, num_sp)
        self.bridge_layer4 = BridgeLayer_new(dims, head, reduction_ratios, num_sp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('Checking bridge')
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B,_,C = bridge4.shape
        outs = []

        sk1 = bridge4[:,:3136,:].reshape(B, 56, 56, C).permute(0,3,1,2) 
        sk2 = bridge4[:,3136:4704,:].reshape(B, 28, 28, C*2).permute(0,3,1,2) 
        sk3 = bridge4[:,4704:5684,:].reshape(B, 14, 14, C*5).permute(0,3,1,2) 
        sk4 = bridge4[:,5684:6076,:].reshape(B, 7, 7, C*8).permute(0,3,1,2) 

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs

class MSTransception(nn.Module):
    def __init__(self, num_classes=9, head_count=8, dil_conv=1, token_mlp_mode="mix_skip", MSViT_config=2, concat='coord', have_bridge='original', use_sa_config = 1,
        sa_ker = 7, Stage_3or4=3, inter='res', num_sp = 1, br_ch_att_list = [True, False, False, False]):#, inception="135"
        super().__init__()
    
        # Encoder
        dims, key_dim, value_dim, layers = [[64, 128, 320, 512], [64, 128, 320, 512], [64, 128, 320, 512], [2, 2, 2, 2]]      
        if use_sa_config==1:
            use_sa_list = [True, True, False]  
        elif use_sa_config==2:
            use_sa_list = [True, False, False]  
        elif use_sa_config==3:
            use_sa_list = [False, False, False] 
        elif use_sa_config==4:
            use_sa_list = [True, True, True]
        else:
            use_sa_list = [True, True, True, False]


        if concat != "cbam" or Stage_3or4 == 4:
            use_sa_list = [True, True, True, False]


        if Stage_3or4 == 4:
            self.backbone = MSViT_4Stages(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers,
                            head_count=head_count, dil_conv=dil_conv, token_mlp=token_mlp_mode, MSViT_config=MSViT_config, concat=concat, use_sa_list=use_sa_list, sa_ker=sa_ker)
        elif Stage_3or4 == 3:
            self.backbone = MSViT(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers,
                            head_count=head_count, dil_conv=dil_conv, token_mlp=token_mlp_mode, MSViT_config=MSViT_config, concat=concat, use_sa_list=use_sa_list, sa_ker=sa_ker)
        else:
            self.backbone = MSViT_casa(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers,
                            head_count=head_count, dil_conv=dil_conv, token_mlp=token_mlp_mode, MSViT_config=MSViT_config, 
                            concat=concat, use_sa_list=use_sa_list, sa_ker=sa_ker, inter=inter)
            
        # self.backbone = MiT_3inception_padding(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers,
        #                     head_count=head_count, dil_conv=dil_conv, token_mlp=token_mlp_mode)

        # Here options:(1) MiT_3inception->3 stages;(2) MiT->4 stages; 
        # (3)MiT_3inception_padding: padding before transformer after patch embedding (follow depthconcat)
        # (4)MiT_3inception_3branches
        # Bridge
        self.reduction_ratios = [1, 2, 4, 8]
        self.have_bridge = have_bridge
        if have_bridge == 'original':
            self.bridge = BridgeBlock_4(64, 1, self.reduction_ratios, br_ch_att_list)
        elif have_bridge == 'sp':
            self.bridge = BridgeBlock_sp(64, 1, self.reduction_ratios, num_sp)
        elif have_bridge == 'para':
            self.bridge = BridgeBlock_para(64, 1, self.reduction_ratios, num_sp)
        else:
            self.bridge = BridgeBlock_4(64, 1, self.reduction_ratios, br_ch_att_list)
        

        # Decoder
        d_base_feat_size = 7 #16 for 512 input size, and 7 for 224
        in_out_chan = [[32, 64, 64, 64],[144, 128, 128, 128],[288, 320, 320, 320],[512, 512, 512, 512]]  # [dim, out_dim, key_dim, value_dim]

        self.decoder_3 = MyDecoderLayer((d_base_feat_size, d_base_feat_size), in_out_chan[3], head_count, 
                                        token_mlp_mode, n_class=num_classes)
        self.decoder_2 = MyDecoderLayer((d_base_feat_size*2, d_base_feat_size*2), in_out_chan[2], head_count,
                                        token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size*4, d_base_feat_size*4), in_out_chan[1], head_count, 
                                        token_mlp_mode, n_class=num_classes) 
        self.decoder_0 = MyDecoderLayer((d_base_feat_size*8, d_base_feat_size*8), in_out_chan[0], head_count,
                                        token_mlp_mode, n_class=num_classes, is_last=True)

        
    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        output_enc = self.backbone(x)
        # print(f"output_enc[0]:{output_enc[0].shape}")
        # print(f"output_enc[1]:{output_enc[1].shape}")
        # print(f"output_enc[2]:{output_enc[2].shape}")
        # print(f"output_enc[3]:{output_enc[3].shape}")
        # return output_enc

        b,c,_,_ = output_enc[3].shape
        #---------------Bridge-----------------------
        if self.have_bridge !="None":
            bridge = self.bridge(output_enc) # list
            b,c,_,_ = bridge[3].shape
            output_enc = bridge
   
        
        #---------------Decoder-------------------------     
        tmp_3 = self.decoder_3(output_enc[3].permute(0,2,3,1).reshape(b,-1,c))
        tmp_2 = self.decoder_2(tmp_3, output_enc[2].permute(0,2,3,1))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0,2,3,1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0,2,3,1))

        return tmp_0
    

if __name__ == "__main__":
    # #call Transception_res

    model = MSTransception(num_classes=9).cuda()
    tmp_0 = model(torch.rand(1, 3, 224, 224).cuda())
    print(tmp_0.shape)    
    summary(model, (1, 3,224,224))