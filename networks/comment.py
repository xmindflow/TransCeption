import torch
import torch.nn as nn

# ResInception 0.71 not working
class MiT_3_ResInception(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip', inception="135"):
        super().__init__()

        self.Hs=[56, 28, 14, 7]
        self.Ws=[56, 28, 14, 7]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        if dil_conv:  
            dilation = 2 
           
            patch_sizes1 = [7, 3, 3, 3]
            dil_padding_sizes1 = [3, 2, 2, 2]    
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
      
        self.conv1_1_s1 = nn.Conv2d((len(inception)+1)*in_dim[0], in_dim[0], 1)
        self.conv1_1_s2 = nn.Conv2d((len(inception)+1)*in_dim[1], in_dim[1], 1)
        self.conv1_1_s3 = nn.Conv2d((len(inception)+1)*in_dim[2], in_dim[2], 1)
        self.conv1_1_s4 = nn.Conv2d((len(inception)+1)*in_dim[3], in_dim[3], 1)

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0])
        
        self.patch_embed2_1 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes1[1], strides[1], dil_padding_sizes1[1],dilation, in_dim[0], in_dim[1])
        # self.patch_embed2_2 = OverlapPatchEmbeddings_fuse(image_size//4, patch_sizes2[1], strides[1], dil_padding_sizes2[1],dilation, in_dim[0], in_dim[1])
        
        
        self.patch_embed3_1 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes1[2], strides[2], dil_padding_sizes1[2],dilation, in_dim[1], in_dim[2])
        # self.patch_embed3_2 = OverlapPatchEmbeddings_fuse(image_size//8, patch_sizes2[2], strides[2], dil_padding_sizes2[2],dilation, in_dim[1], in_dim[2])

        self.patch_embed4_1 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes1[3], strides[3], dil_padding_sizes1[3],dilation, in_dim[2], in_dim[3])
        # self.patch_embed4_2 = OverlapPatchEmbeddings_fuse(image_size//16, patch_sizes2[3], strides[3], dil_padding_sizes2[3],dilation, in_dim[2], in_dim[3])
        
        # inception branch
        multiResBlock = {
                        '15': MultiResBlock_15,
                        '13': MultiResBlock_13,
                        '1': MultiResBlock_1,
                        '3': MultiResBlock_3,
                        '5': MultiResBlock_5,
                        }
        
       
        self.resInception2_2 = multiResBlock[inception](in_dim[0],in_dim[1],branch=1,downsample=strides[1],alpha=1)
        self.resInception3_2 = multiResBlock[inception](in_dim[1],in_dim[2],branch=1,downsample=strides[2],alpha=1)
        self.resInception4_2 = multiResBlock[inception](in_dim[2],in_dim[3],branch=1,downsample=strides[3],alpha=1)
        

        # transformer encoder
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList([
            EfficientTransformerBlockFuse_res(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp)
        for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList([
            EfficientTransformerBlockFuse_res(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp)
        for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(in_dim[2])

        self.block4 = nn.ModuleList([
            EfficientTransformerBlockFuse_res(in_dim[3], key_dim[3], value_dim[3], head_count, token_mlp)
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

      

        # merge 2
        # print("-------EN: Stage 2------\n\n")
        x1, H1, W1 = self.patch_embed2_1(x)
        H2 = H1
        W2 = W1
        # print("\n S2: H1:{}, H2:{}".format(H1,H2))
        _, nfx1_len, _ = x1.shape
        x2 = self.resInception2_2(x)
        _, nfx2_len, _ = x2.shape
        # print("\n x2 shape:", x2.shape)
        nfx_cat = torch.cat((x1,x2),1)

        # stage 2

        for blk in self.block2:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        tx = self.norm2(nfx_cat)
        # The mlp has been passed in blk, so next just split the sequence and 
        # reshape to spatial dimension
        b,tx_len,_ = tx.shape
        # z_total = []
        map_mx_total = []
        for nz in range(int(tx_len/nfx1_len)):
            z = tx[:, nz*nfx1_len:(nz+1)*nfx1_len, :]
            # z_total.append(z)
            # print( z.shape)
            map_mx = z.reshape(b,H1,W1,-1)
            map_mx = map_mx.permute(0,3,1,2)
            # print( "\nmap_mx: ",map_mx.shape)
            map_mx_total.append(map_mx)

        cat_maps = torch.cat(map_mx_total,1)
        x = self.conv1_1_s2(cat_maps)
        outs.append(x)

        
       # merge 3
        x1, H1, W1 = self.patch_embed3_1(x)
        H2 = H1
        W2 = W1
        # print("\n S3: H1:{}, H2:{}".format(H1,H2))
        _, nfx1_len, _ = x1.shape
        x2 = self.resInception3_2(x)
        _, nfx2_len, _ = x2.shape
        nfx_cat = torch.cat((x1,x2),1)

        # stage 3
        for blk in self.block3:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        tx = self.norm3(nfx_cat)

        b,tx_len,_ = tx.shape
        # z_total = []
        map_mx_total = []
        for nz in range(int(tx_len/nfx1_len)):
            z = tx[:, nz*nfx1_len:(nz+1)*nfx1_len, :]
            # z_total.append(z)
            # print( z.shape)
            map_mx = z.reshape(b,H1,W1,-1)
            map_mx = map_mx.permute(0,3,1,2)
            # print( "\nmap_mx: ",map_mx.shape)
            map_mx_total.append(map_mx)

        cat_maps = torch.cat(map_mx_total,1)
        x = self.conv1_1_s3(cat_maps)
        outs.append(x)


        # merge 4
      
        x1, H1, W1 = self.patch_embed4_1(x)
        H2 = H1
        W2 = W1
        # print("\n S4: H1:{}, H2:{}".format(H1,H2))
        _, nfx1_len, _ = x1.shape
        x2 = self.resInception4_2(x)
        _, nfx2_len, _ = x2.shape
        nfx_cat = torch.cat((x1,x2),1)

        # stage 4
        for blk in self.block4:
            nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
        tx = self.norm4(nfx_cat)
        b,tx_len,_ = tx.shape
        # z_total = []
        map_mx_total = []
        for nz in range(int(tx_len/nfx1_len)):
            z = tx[:, nz*nfx1_len:(nz+1)*nfx1_len, :]
            # z_total.append(z)
            # print( z.shape)
            map_mx = z.reshape(b,H1,W1,-1)
            map_mx = map_mx.permute(0,3,1,2)
            # print( "\nmap_mx: ",map_mx.shape)
            map_mx_total.append(map_mx)

        cat_maps = torch.cat(map_mx_total,1)
        x = self.conv1_1_s4(cat_maps)
        outs.append(x)


        return outs
  
# not effective
class Eff_FactorAtt_ConvRelPosEnc(nn.Module):
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
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reprojection = nn.Conv2d(dim, dim, 1)
        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):

        # print('inside the Eff_FactorAtt')
        # EfficientAttention(in_channels=in_dim, key_channels=key_dim,value_channels=value_dim, head_count=1)
        B, N, C = x.shape
        H,W = size
        # x = Rearrange('b (h w) d -> b d h w', h=H, w=W)(x)
        qkv = self.qkv(x).reshape(B,N,3,C).permute(2,0,3,1) #B,N,3C->B,N,3,C->3,B,C,N
        q, k, v = qkv[0], qkv[1], qkv[2]

        head_key_channels = self.head_dim
        head_value_channels = self.head_dim

        attended_values = []
        for i in range(self.num_heads):
            key = F.softmax(k[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            
            query = F.softmax(q[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
                        
            value = v[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]  

            context = key @ value.transpose(1, 2) # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(B, head_value_channels, H, W) # n*dv            
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values) #[B,C,H,W]
        attention = Rearrange('b d h w -> b (h w) d')(attention)
        # Generate Q, K, V.
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        #     .contiguous()
        # )  # Shape: [3, B, h, N, Ch].
        # q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].
#-----------
        # # Factorized attention.
        # k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        # k_softmax_T_dot_v = einsum(
        #     "b h n k, b h n v -> b h k v", k_softmax, v
        # )  # Shape: [B, h, Ch, Ch].
        # factor_att = einsum(
        #     "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        # )  # Shape: [B, h, N, Ch].

        # # Convolutional relative position encoding.
        # crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].

        # # Merge and reshape.
        # x = self.scale * factor_att + crpe
        # x = (
        #     x.transpose(1, 2).reshape(B, N, C).contiguous()
        # )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # # Output projection.
        # x = self.proj(x)
        # x = self.proj_drop(x)

        return attention

# not effective
class CAM_Module_new(nn.Module):
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
        proj_query = x.reshape(m_batchsize, C, -1) #b c n
        # print(f'Shape of query: {proj_query.shape}')
        proj_key = x.reshape(m_batchsize, C, -1).permute(0, 2, 1) #b n c 
        # print(f'Shape of key: {proj_key.shape}') 
        energy = torch.matmul(proj_query, proj_key) # b c c
        # print(f'Shape of energy: {energy.shape}')
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy# The larger dependency of the channels, the lower energy_new value/ used to emphasizing the different information
        
        # min_energy_0 = torch.min(energy, -1, keepdim=True)[0].expand_as(energy)
        # energy_new = energy - min_energy_0
        # print(f'energy_new: {energy_new.shape}')
        
        attention = self.softmax(energy_new)
        # print(f'Shape of attention: {attention[0]}\n')
        proj_value = x.reshape(m_batchsize, C, -1) #b c n
        # print(f'Shape of proj_value: {proj_value.shape}')

        out = torch.matmul(attention, proj_value)# can be replace by torch.matmul b c n
        # print(f'Shape of out: {out.shape}\n')
        out = out.reshape(m_batchsize, C, numpath, height, width)


#         logging.debug('cam device: {}, {}'.format(out.device, self.gamma.device))
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out
# Try to add the bridge

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
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)
            
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)


        return out

class BridgeLayer_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
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



class BridegeBlock_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_4(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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



#cbam
