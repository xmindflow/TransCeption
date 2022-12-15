from cmath import sqrt
import torch
import torch.nn as nn
# from networks.segformer import *
# For jupyter notebook below
from .EffSegformer import *
# from yiwei_gitlab.EffFormer.networks.inception import *
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

# From MISSFormer.py class BridgeLayer_4
# From Transception.py line83 forward part
# FromEfficientAttention to FuseEfficientAttention


class FuseEfficientAttention(nn.Module):
    """
        input  -> x:[B, N, D]
        output ->   [B, N, D]
    
        in_channels:    int -> Embedding Dimension  d
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
        
        Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """
    
    def __init__(self, in_channels, key_channels, value_channels, head_count):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        # print(f'in fuse attention:{head_count}')
        self.value_channels = value_channels

        self.keys = nn.Linear(in_channels, key_channels, bias=True) 
        self.queries = nn.Linear(in_channels, key_channels, bias=True)
        self.values = nn.Linear(in_channels, value_channels,bias=True)
        self.reprojection = nn.Linear(value_channels, in_channels)

        
    def forward(self, input_):
        b, n, _ = input_.size()
        # print("\nb is {}, n is {}\n".format(b, n))
        # print("\nin_channels is {}, key_channels is {}\n".format( self.in_channels, self.key_channels))
        # n, _,  = input_.size()
        keys = self.keys(input_)#B N D
        keys = keys.reshape((b, self.key_channels, n))#B dk N
        queries = self.queries(input_).reshape(b, self.key_channels, n)#b dk n
        values = self.values(input_).reshape((b, self.value_channels, n))# b dv n
        
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
            
            context = key @ value.transpose(1, 2) # dk*dv B dk N* B N dv=B dk dv
            attended_value = (context.transpose(1, 2) @ query).reshape(b, head_value_channels, n) 
            # (b dv dk @ b dk n)->b dv n         
            attended_values.append(attended_value)
                
        aggregated_values = torch.cat(attended_values, dim=1).permute(0,2,1)
        #b n h*dv
        attention = self.reprojection(aggregated_values)# b n d

        return attention


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
        #--------
        # Q K V: [B,Ch,N]
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
    

class EfficientTransformerBlockFuse(nn.Module):
    """
        Input  -> x (Size: (b, n1+n2, d)), nfx1, nfx2
        Output -> (b, (H*W), d)
    """
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix'):
        super().__init__()
    
        self.norm1 = nn.LayerNorm(in_dim)
        # self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim,
                                    #    value_channels=value_dim, head_count=1)

        self.attn = FuseEfficientAttention(in_channels=in_dim, key_channels=key_dim,value_channels=value_dim, head_count=head_count)
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp1 = MixFFN(in_dim, int(in_dim*4))
            self.mlp2 = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim*4)) 
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim*4)) 
            # self.mlp2 = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim*4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim*4))

    def forward(self, x: torch.Tensor, nfx1_len, nfx2_len, H1, W1,H2,W2) -> torch.Tensor:
        _,x_len,_=x.shape
        norm_1 = self.norm1(x)
        # norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
        attn = self.attn(norm_1)
        # attn = Rearrange('b d h w -> b (h w) d')(attn)
        
        tx = x + attn
        # 10-14 adjust the order of norm2 and split
        # tx = self.norm2(tx) 
        # z1 = tx[:, :nfx1_len, :]
        # z2 = tx[:, nfx1_len:, :]

        # changed order: first split and skip connection, 
        # then norm to mlp
        if x_len == nfx1_len + nfx2_len:
            z1 = tx[:, :nfx1_len, :]
            z2 = tx[:, nfx1_len:, :]
            mx1 = z1 + self.mlp1(self.norm2(z1), H1, W1)
            mx2 = z2 + self.mlp2(self.norm2(z2), H2, W2)

            mx = torch.cat((mx1, mx2),1)
        else:
            z1 = tx[:, :nfx1_len, :]
            z2 = tx[:, nfx1_len:(nfx1_len+nfx2_len), :]
            z3 = tx[:, (nfx1_len+nfx2_len):, :]
            # print("nfx1_len:{} nfx2_len:{} nfx3_len{}".format(nfx1_len,nfx2_len,x_len - nfx1_len -nfx2_len))
            H3 = int(sqrt(x_len - nfx1_len - nfx2_len).real)
            # print("H3: ",H3)
            W3 = H3
            mx1 = z1 + self.mlp1(self.norm2(z1), H1, W1)
            mx2 = z2 + self.mlp2(self.norm2(z2), H2, W2)
            mx3 = z3 + self.mlp2(self.norm2(z3), H3, W3)

            mx = torch.cat((mx1, mx2, mx3),1)

       
        return mx
    
class EfficientTransformerBlockFuse_res(nn.Module):
    """
        Input  -> x (Size: (b, n1+n2, d)), nfx1, nfx2
        Output -> (b, (H*W), d)
    """
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        # self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim,
                                    #    value_channels=value_dim, head_count=1)

        self.attn = FuseEfficientAttention(in_channels=in_dim, key_channels=key_dim,value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(in_dim, int(in_dim*4))
            # self.mlp2 = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim*4)) 
            # self.mlp2 = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim*4))
            # self.mlp2 = MLP_FFN(in_dim, int(in_dim*4))

    def forward(self, x: torch.Tensor, nfx1_len, nfx2_len, H1, W1,H2,W2) -> torch.Tensor:
        
        norm_1 = self.norm1(x)

        # norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
        attn = self.attn(norm_1)
        # attn = Rearrange('b d h w -> b (h w) d')(attn)
        
        tx = x + attn
        _,tx_len,_ = tx.shape
        # 10-14 adjust the order of norm2 and split
        # tx = self.norm2(tx) 
        # z1 = tx[:, :nfx1_len, :]
        # z2 = tx[:, nfx1_len:, :]

        # changed order: first split and skip connection, 
        # then norm to mlp
        # z_total = []
        mx = []
        for nz in range(int(tx_len/nfx1_len)):
            z = tx[:, nz*nfx1_len:(nz+1)*nfx1_len, :]
            # z_total.append(z)
            # print( z.shape)
            mx_nz = z+ self.mlp(z, H1, W1)
            mx.append(mx_nz)

        mx = torch.cat(mx,1)
        return mx
    
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

    

class MiT_3inception(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip', concat='original'):
        super().__init__()

        self.Hs=[56, 28, 14, 7]
        self.Ws=[56, 28, 14, 7]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        if dil_conv:  
            dilation = 2 
            # # conv 5
            # patch_sizes1 = [7, 3, 3, 3]
            # patch_sizes2 = [5, 1, 1, 1]                
            # dil_padding_sizes1 = [3, 0, 0, 0]
            # dil_padding_sizes2 = [3, 0, 0, 0]
            # # conv 3
            # patch_sizes1 = [7, 3, 3, 3]
            # dil_padding_sizes1 = [3, 0, 0, 0]    
            # patch_sizes2 = [3, 1, 1, 1]
            # dil_padding_sizes2 = [1, 0, 0, 0]
            # conv 1
            patch_sizes1 = [7, 3, 3, 3]
            dil_padding_sizes1 = [3, 0, 0, 0]    
            patch_sizes2 = [1, 1, 1, 1]
            dil_padding_sizes2 = [0, 0, 0, 0]
        else:
            dilation = 1
            patch_sizes1 = [7, 3, 3, 3]
            patch_sizes2 = [5, 1, 1, 1]
            dil_padding_sizes1 = [3, 1, 1, 1]
            # dil_padding_sizes2 = [3, 0, 0, 0]
            dil_padding_sizes2 = [1, 0, 0, 0]


        # 1 by 1 convolution to alter the dimension
        self.conv1_1_s1 = nn.Conv2d(2*in_dim[0], in_dim[0], 1)
        self.conv1_1_s2 = nn.Conv2d(2*in_dim[1], in_dim[1], 1)
        self.conv1_1_s3 = nn.Conv2d(2*in_dim[2], in_dim[2], 1)
        self.conv1_1_s4 = nn.Conv2d(2*in_dim[3], in_dim[3], 1)

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0])
        
        self.patch_embed2_1 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes1[1], strides[1], dil_padding_sizes1[1],dilation, in_dim[0], in_dim[1])
        self.patch_embed2_2 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes2[1], strides[1], dil_padding_sizes2[1],dilation, in_dim[0], in_dim[1])

        self.patch_embed3_1 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes1[2], strides[2], dil_padding_sizes1[2],dilation, in_dim[1], in_dim[2])
        self.patch_embed3_2 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes2[2], strides[2], dil_padding_sizes2[2],dilation, in_dim[1], in_dim[2])

        self.patch_embed4_1 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes1[3], strides[3], dil_padding_sizes1[3],dilation, in_dim[2], in_dim[3])
        self.patch_embed4_2 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes2[3], strides[3], dil_padding_sizes2[3],dilation, in_dim[2], in_dim[3])
        
        # transformer encoder
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp)
        for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp)
        for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(in_dim[2])

        self.block4 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[3], key_dim[3], value_dim[3], head_count, token_mlp)
        for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(in_dim[3])

        self.concat = concat
        self.sk_concat2 = SK_Block(in_dim[1], in_dim[1], num_path=2,reduction=16)
        self.sk_concat3 = SK_Block(in_dim[2], in_dim[2], num_path=2,reduction=16)
        self.sk_concat4 = SK_Block(in_dim[3], in_dim[3], num_path=2,reduction=16)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []
        num_path = 2

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

      

        # stage 2
        # print("-------EN: Stage 2------\n\n")
        x1, H1, W1 = self.patch_embed2_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed2_2(x)
        _, nfx2_len, _ = x2.shape
        nfx_cat = torch.cat((x1,x2),1)

        for blk in self.block2:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm2(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len:, :]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1)
        map_mx2 = mx2.reshape(b,H2,W2,-1)
        map_mx1 = map_mx1.permute(0,3,1,2)
        map_mx2 = map_mx2.permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[1], self.Ws[1]])
        if self.concat == 'original':      
            cat_maps = torch.cat((map_mx1, map_mx2),1)
            x = self.conv1_1_s2(cat_maps)
        else:
            in_sk = []
            in_sk.append(map_mx1)
            in_sk.append(map_mx2)
            x = self.sk_concat2(in_sk)
           
        outs.append(x)


            

        # stage 3
       
        x1, H1, W1 = self.patch_embed3_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed3_2(x)
        _, nfx2_len, _ = x2.shape
        nfx_cat = torch.cat((x1,x2),1)

        for blk in self.block3:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm3(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len: :]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1)
        map_mx2 = mx2.reshape(b,H2,W2,-1)
        map_mx1 = map_mx1.permute(0,3,1,2)
        map_mx2 = map_mx2.permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[2], self.Ws[2]])
        if self.concat == 'original':      
            cat_maps = torch.cat((map_mx1, map_mx2),1)
            x = self.conv1_1_s3(cat_maps)
        else:
            in_sk = []
            in_sk.append(map_mx1)
            in_sk.append(map_mx2)
            x = self.sk_concat3(in_sk)
           
        outs.append(x)

        # stage 4
      
        x1, H1, W1 = self.patch_embed4_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed4_2(x)
        _, nfx2_len, _ = x2.shape
        nfx_cat = torch.cat((x1,x2),1)

        for blk in self.block4:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm4(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len: :]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1)
        map_mx2 = mx2.reshape(b,H2,W2,-1)
        map_mx1 = map_mx1.permute(0,3,1,2)
        map_mx2 = map_mx2.permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[3], self.Ws[3]])
        if self.concat == 'original':      
            cat_maps = torch.cat((map_mx1, map_mx2),1)
            x = self.conv1_1_s4(cat_maps)
        else:
            in_sk = []
            in_sk.append(map_mx1)
            in_sk.append(map_mx2)
            x = self.sk_concat4(in_sk)
           
        outs.append(x)

        return outs
    
class MiT_3inception_3branches(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip'):
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

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0])
        
        self.patch_embed2_1 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes1[1], strides[1], dil_padding_sizes1[1],dilation, in_dim[0], in_dim[1])
        self.patch_embed2_2 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes2[1], strides[1], dil_padding_sizes2[1],dilation, in_dim[0], in_dim[1])
        self.patch_embed2_3 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes3[1], strides[1], dil_padding_sizes3[1],dilation, in_dim[0], in_dim[1])

        self.patch_embed3_1 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes1[2], strides[2], dil_padding_sizes1[2],dilation, in_dim[1], in_dim[2])
        self.patch_embed3_2 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes2[2], strides[2], dil_padding_sizes2[2],dilation, in_dim[1], in_dim[2])
        self.patch_embed3_3 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes3[2], strides[2], dil_padding_sizes3[2],dilation, in_dim[1], in_dim[2])

        self.patch_embed4_1 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes1[3], strides[3], dil_padding_sizes1[3],dilation, in_dim[2], in_dim[3])
        self.patch_embed4_2 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes2[3], strides[3], dil_padding_sizes2[3],dilation, in_dim[2], in_dim[3])
        self.patch_embed4_3 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes3[3], strides[3], dil_padding_sizes3[3],dilation, in_dim[2], in_dim[3])
        
        # transformer encoder
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp)
        for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp)
        for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(in_dim[2])

        self.block4 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[3], key_dim[3], value_dim[3], head_count, token_mlp)
        for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(in_dim[3])
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

      

        # stage 2
        # print("-------EN: Stage 2------\n\n")
        x1, H1, W1 = self.patch_embed2_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed2_2(x)
        _, nfx2_len, _ = x2.shape
        x3, H3, W3 = self.patch_embed2_3(x)
        # _, nfx3_len, _ = x3.shape
        nfx_cat = torch.cat((x1,x2,x3),1)

        for blk in self.block2:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm2(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len:(nfx1_len+nfx2_len), :]
        mx3 = nfx_cat[:, (nfx1_len+nfx2_len):,:]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1).permute(0,3,1,2)
        map_mx2 = mx2.reshape(b,H2,W2,-1).permute(0,3,1,2)
        map_mx3 = mx3.reshape(b,H3,W3,-1).permute(0,3,1,2)
        # map_mx1 = map_mx1.permute(0,3,1,2)
        # map_mx2 = map_mx2.permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[1], self.Ws[1]])
        map_mx2 = F.interpolate(map_mx2,[self.Hs[1], self.Ws[1]])
        map_mx3 = F.interpolate(map_mx3,[self.Hs[1], self.Ws[1]])
        cat_maps = torch.cat((map_mx1, map_mx2, map_mx3),1)
        x = self.conv1_1_s2(cat_maps)
        outs.append(x)

        # stage 3
       
        x1, H1, W1 = self.patch_embed3_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed3_2(x)
        _, nfx2_len, _ = x2.shape
        x3, H3, W3 = self.patch_embed3_3(x)
        nfx_cat = torch.cat((x1,x2,x3),1)

        for blk in self.block3:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm3(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len:(nfx1_len+nfx2_len), :]
        mx3 = nfx_cat[:, (nfx1_len+nfx2_len):,:]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1).permute(0,3,1,2)
        map_mx2 = mx2.reshape(b,H2,W2,-1).permute(0,3,1,2)
        map_mx3 = mx3.reshape(b,H3,W3,-1).permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[2], self.Ws[2]])
        map_mx2 = F.interpolate(map_mx2,[self.Hs[2], self.Ws[2]])
        map_mx3 = F.interpolate(map_mx3,[self.Hs[2], self.Ws[2]])
        cat_maps = torch.cat((map_mx1, map_mx2,map_mx3),1)
        x = self.conv1_1_s3(cat_maps)
        outs.append(x)

        # stage 4
      
        x1, H1, W1 = self.patch_embed4_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed4_2(x)
        _, nfx2_len, _ = x2.shape
        x3, H3, W3 = self.patch_embed4_3(x)
        nfx_cat = torch.cat((x1,x2,x3),1)


        for blk in self.block4:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm4(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len:(nfx1_len+nfx2_len), :]
        mx3 = nfx_cat[:, (nfx1_len+nfx2_len):,:]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1).permute(0,3,1,2)
        map_mx2 = mx2.reshape(b,H2,W2,-1).permute(0,3,1,2)
        map_mx3 = mx3.reshape(b,H3,W3,-1).permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[3], self.Ws[3]])
        map_mx2 = F.interpolate(map_mx2,[self.Hs[3], self.Ws[3]])
        map_mx3 = F.interpolate(map_mx3,[self.Hs[3], self.Ws[3]])
        # map_mx1 = F.interpolate(map_mx1,[self.Hs[3], self.Ws[3]])
        cat_maps = torch.cat((map_mx1, map_mx2, map_mx3),1)
        x = self.conv1_1_s4(cat_maps)
        outs.append(x)

        return outs
    

    
# Encoder MiT with 4 inception modules 
class MiT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp='mix_skip'):
        super().__init__()

        self.Hs=[56, 28, 14, 7]
        self.Ws=[56, 28, 14, 7]

        patch_sizes = [7, 3, 3, 3]
        patch_sizes1 = [7, 3, 3, 3]
        patch_sizes2 = [5, 1, 1, 1]

        strides = [4, 2, 2, 2]
        # padding_sizes = [3, 1, 1, 1]
        dil_padding_sizes1 = [3, 0, 0, 0]
        dil_padding_sizes2 = [3, 0, 0, 0]

        # 1 by 1 convolution to alter the dimension
        self.conv1_1_s1 = nn.Conv2d(2*in_dim[0], in_dim[0], 1)
        self.conv1_1_s2 = nn.Conv2d(2*in_dim[1], in_dim[1], 1)
        self.conv1_1_s3 = nn.Conv2d(2*in_dim[2], in_dim[2], 1)
        self.conv1_1_s4 = nn.Conv2d(2*in_dim[3], in_dim[3], 1)

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1_1 = OverlapPatchEmbeddings_fuse(image_size, patch_sizes1[0], strides[0], dil_padding_sizes1[0], 3, in_dim[0])
        self.patch_embed1_2 = OverlapPatchEmbeddings_fuse(image_size, patch_sizes2[0], strides[0], dil_padding_sizes2[0], 3, in_dim[0])

        self.patch_embed2_1 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes1[1], strides[1], dil_padding_sizes1[1],in_dim[0], in_dim[1])
        self.patch_embed2_2 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes2[1], strides[1], dil_padding_sizes2[1],in_dim[0], in_dim[1])

        self.patch_embed3_1 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes1[2], strides[2], dil_padding_sizes1[2],in_dim[1], in_dim[2])
        self.patch_embed3_2 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes2[2], strides[2], dil_padding_sizes2[2],in_dim[1], in_dim[2])

        self.patch_embed4_1 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes1[3], strides[3], dil_padding_sizes1[3],in_dim[2], in_dim[3])
        self.patch_embed4_2 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes2[3], strides[3], dil_padding_sizes2[3],in_dim[2], in_dim[3])
        
        # transformer encoder
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlockFuse(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp)
        for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp)
        for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(in_dim[2])

        self.block4 = nn.ModuleList([
            EfficientTransformerBlockFuse(in_dim[3], key_dim[3], value_dim[3], head_count, token_mlp)
        for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(in_dim[3])
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        
        x1, H1, W1 = self.patch_embed1_1(x)
        _, nfx1_len, d1 = x1.shape
        x2, H2, W2 = self.patch_embed1_2(x)
        _, nfx2_len, d2 = x2.shape
        nfx_cat = torch.cat((x1,x2),1)
        

        for blk in self.block1:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)

        # print("nfx_cat shape {}".format(nfx_cat.shape))
        nfx_cat = self.norm1(nfx_cat)
        # print("nfx_cat shape {}".format(nfx_cat.shape))
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len:, :]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1)
        map_mx2 = mx2.reshape(b,H2,W2,-1)
        map_mx1 = map_mx1.permute(0,3,1,2)
        map_mx2 = map_mx2.permute(0,3,1,2)
        # print("map_mx1 shape:{} \n map_mx2 shape:{}".format(map_mx1.shape, map_mx2.shape))
        map_mx1 = F.interpolate(map_mx1,[self.Hs[0], self.Ws[0]])
        cat_maps = torch.cat((map_mx1, map_mx2),1)
        # print("cat_maps shape:{}".format(cat_maps.shape))
        x = self.conv1_1_s1(cat_maps)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        # print("-------EN: Stage 2------\n\n")
        x1, H1, W1 = self.patch_embed2_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed2_2(x)
        _, nfx2_len, _ = x2.shape
        nfx_cat = torch.cat((x1,x2),1)

        for blk in self.block2:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm2(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len:, :]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1)
        map_mx2 = mx2.reshape(b,H2,W2,-1)
        map_mx1 = map_mx1.permute(0,3,1,2)
        map_mx2 = map_mx2.permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[1], self.Ws[1]])
        cat_maps = torch.cat((map_mx1, map_mx2),1)
        x = self.conv1_1_s2(cat_maps)
        outs.append(x)

        # stage 3
       
        x1, H1, W1 = self.patch_embed3_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed3_2(x)
        _, nfx2_len, _ = x2.shape
        nfx_cat = torch.cat((x1,x2),1)

        for blk in self.block3:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm3(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len:, :]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1)
        map_mx2 = mx2.reshape(b,H2,W2,-1)
        map_mx1 = map_mx1.permute(0,3,1,2)
        map_mx2 = map_mx2.permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[2], self.Ws[2]])
        cat_maps = torch.cat((map_mx1, map_mx2),1)
        x = self.conv1_1_s3(cat_maps)
        outs.append(x)

        # stage 4
      
        x1, H1, W1 = self.patch_embed4_1(x)
        _, nfx1_len, _ = x1.shape
        x2, H2, W2 = self.patch_embed4_2(x)
        _, nfx2_len, _ = x2.shape
        nfx_cat = torch.cat((x1,x2),1)

        for blk in self.block4:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        nfx_cat = self.norm4(nfx_cat)
        mx1 = nfx_cat[:, :nfx1_len, :]
        mx2 = nfx_cat[:, nfx1_len:, :]
        b, _, _ = mx1.shape
        map_mx1 = mx1.reshape(b,H1,W1,-1)
        map_mx2 = mx2.reshape(b,H2,W2,-1)
        map_mx1 = map_mx1.permute(0,3,1,2)
        map_mx2 = map_mx2.permute(0,3,1,2)
        map_mx1 = F.interpolate(map_mx1,[self.Hs[3], self.Ws[3]])
        cat_maps = torch.cat((map_mx1, map_mx2),1)
        x = self.conv1_1_s4(cat_maps)
        outs.append(x)

        return outs
    
# Decoder    
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
    
    
class Transception(nn.Module):
    def __init__(self, num_classes=9, head_count=1, dil_conv=1, token_mlp_mode="mix_skip", concat='original'):#, inception="135"
        super().__init__()
    
        # Encoder
        dims, key_dim, value_dim, layers = [[64, 128, 320, 512], [64, 128, 320, 512], [64, 128, 320, 512], [2, 2, 2, 2]]        
        self.backbone = MiT_3inception(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers,
                            head_count=head_count, dil_conv=dil_conv, token_mlp=token_mlp_mode, concat=concat)
        # self.backbone = MiT_3inception_padding(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers,
        #                     head_count=head_count, dil_conv=dil_conv, token_mlp=token_mlp_mode, concat=concat)

        # Here options:(1) MiT_3inception->3 stages;(2) MiT->4 stages; 
        # (3)MiT_3inception_padding: padding before transformer after patch embedding (follow depthconcat)
        # (4)MiT_3inception_3branches
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
        # print('output_enc 0: {}'.format(output_enc[0].shape))
        # print('output_enc 1: {}'.format(output_enc[1].shape))
        # print('output_enc 2: {}'.format(output_enc[2].shape))
        # print('output_enc 3: {}'.format(output_enc[3].shape))

        b,c,_,_ = output_enc[3].shape

        #---------------Decoder-------------------------     
        tmp_3 = self.decoder_3(output_enc[3].permute(0,2,3,1).view(b,-1,c))
        tmp_2 = self.decoder_2(tmp_3, output_enc[2].permute(0,2,3,1))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0,2,3,1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0,2,3,1))

        return tmp_0
    
# if __name__ == "__main__":
#     #call Transception_res
#     model = Transception(num_classes=9, head_count=8, dil_conv = 1, token_mlp_mode="mix_skip", concat='sk')

#     print(model(torch.rand(1, 3, 224, 224)).shape)
    