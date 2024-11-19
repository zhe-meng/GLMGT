import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC

        return x
class PatchEmbedLayerSpa(nn.Module):
    def __init__(self, patch_size=11, in_dim=3, embedding_dims=32, is_first_layer=False):
        super().__init__()
        if is_first_layer:                                    
            patch_size = 1
            in_dim = embedding_dims

        patch_size = to_2tuple(patch_size) # 1,1 or 2,2
        self.patch_size = patch_size

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(1, 1, 5), stride=1, padding=(0, 0, 2)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*embedding_dims*3, out_channels=embedding_dims, kernel_size=(1, 1)),
            nn.BatchNorm2d(embedding_dims),
            nn.GELU(),
        )

    def forward(self, x):
        # _, _, H, W = x.shape
        # out_H, out_W = H // self.patch_size[0], W // self.patch_size[1]

        input =  x
        x = x.unsqueeze(1)
        x1 = self.conv3d_1(x)
        x2 = self.conv3d_2(x)
        x3 = self.conv3d_3(x)
        x = torch.cat((x1, x2, x3), dim=1) #  
        x = rearrange(x, 'b c h w y -> b c y h w ')
        x = rearrange(x, 'b c y h w -> b (c y) h w ')

        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b  h w c')
        x = x + input
   
        # return x, (out_H, out_W)
        return x
    
class MGEFE(nn.Module):
    def __init__(self, patch_size,dim, num_heads, bias):
        super(MGEFE, self).__init__()
        self.multi = PatchEmbedLayerSpa(patch_size=patch_size,embedding_dims=dim)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)  #原始
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)

        # self.qkv_dwconv = nn.Conv2d(dim, dim*3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.multi(x)
        b,h,w,c, = x.shape
        x = rearrange(x, 'b h w c -> b c h w')

        # 升维，卷积，分块得到qkv
        qkv = self.qkv_dwconv(self.qkv(x))

        # qkv = self.qkv_dwconv(x) #无1×1卷积
        q,k,v = qkv.chunk(3, dim=1)   
        # 维度变化 [B, C, H, W] ==> [B, head, C/head, HW] 
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # [B, head, C/head, HW] * [B, head, HW, C/head] * [head, 1, 1] ==> [B, head, C/head, C/head]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # [B, head, C/head, C/head] * [B, head, C/head, HW] ==> [B, head, C/head, HW]
        out = (attn @ v)
        
        # [B, head, C/head, HW] ==> [B, head, C/head, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = rearrange(x, 'b c h w -> b h w c ')
        return out


class MGAFE(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MGAFE, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) # 可学习的缩放参数，应用softmax函数之前控制(K和(Q)的点积的大小)

        # 设置三个不同尺度的卷积层，用于提取不同尺度的特征
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.qkv_3 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=bias)
 

        # 使用深度可分离卷积提取不同尺度的特征
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,h,w,c, = x.shape
        x = rearrange(x, 'b h w c -> b c h w')

        # 提取三个不同尺度的特征
        qkv_1 = self.qkv_1(x)
        qkv_2 = self.qkv_2(x)
        qkv_3 = self.qkv_3(x)

        # 合并三个不同尺度的特征
        qkv = torch.cat([qkv_1, qkv_2, qkv_3], dim=1)

   
        
        # 升维，卷积，分块得到qkv
        qkv = self.qkv_dwconv(qkv)
        q,k,v = qkv.chunk(3, dim=1)  

        # 维度变化 [B, C, H, W] ==> [B, head, HW/head, c ] 
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # [B, head, HW/head, c] * [B, head,  c, HW/head] * [head, 1, 1] ==> [B, head, HW/head, HW/head]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # [B, head, HW/head, HW/head] * [B, head, HW/head, c ]  ==> [B, head,HW/head, C]
        out = (attn @ v)
        
        # B, head,HW/head, C] ==> [B, head, C/head, H, W]
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = rearrange(x, 'b c h w -> b h w c ')
        return out

## GFFM module 
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        # _, t, _ = x.size()
        # H = W = int(math.sqrt(t))
        # x = rearrange(x, 'n (h w) c -> n c h w ', h=H, w=W)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        
        # x = rearrange(x, 'n c h w -> n (h w) c')
        return x


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                       num_heads=8,  qk_dim=None,  mlp_ratio=4, mlp_dwconv=False,
                       before_attn_dwconv=3, ffn_expansion_factor=2,pre_norm=True,patch_size=11):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)# depth_wise CNN  DCPE
    
        else:
   
            self.pos_embed = nn.Parameter(torch.randn(1, dim, 1, 1))  # Assuming 2D positional encoding

        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
        self.attn = MGEFE(patch_size=patch_size,dim=dim,num_heads=num_heads,bias=False)  #多尺度MDTA 注意力图：c*c
        self.attn4 = MGAFE(dim=dim,num_heads=num_heads,bias=False) #在MDTA基础改QKV形状去改注意力图：HW*HW，加多尺度
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 DWConv(int(mlp_ratio*dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.FeedForward = FeedForward(dim,ffn_expansion_factor = ffn_expansion_factor,bias = True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
            

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.InvertedResidual(self.norm2(x))) # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn4(self.norm3(x)))# (N, H, W, C)#串联
                x = x + self.drop_path(self.attn(self.norm1(x)))# (N, H, W, C)
                x = self.norm2(x)
                x = x.permute(0, 3, 2, 1)
                x = self.FeedForward(x)  #(GDFN)
                x = x.permute(0, 2, 3, 1)
                x = x + self.drop_path(x) # (N, H, W, C)  GFFN

        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x))) # (N, H, W, C)
                # x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.InvertedResidual(x))) # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x))) # (N, H, W, C)
                # x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.InvertedResidual(x))) # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x



class GLMGT(nn.Module):
    def __init__(self, in_chans=200, num_classes=1000, patch_size=11, depth=[1], embed_dim=[32],
                 num_head=8, head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., 
                 ######## 
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 pre_norm=True,
                 before_attn_dwconv=3,
                 ffn_expansion_factor=5,
                 auto_pad=False,
                 mlp_dwconv=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ Dimensionality Reduction layers (patch embeddings) ######################
        self.dimensreduct_layers = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
            nn.GELU()
        )
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        num_head= num_head
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))] 
        cur = 0
  
        stage = nn.Sequential(
            *[Block(dim=embed_dim[0], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value,
                    num_heads=num_head,
                    qk_dim=qk_dims[0],
                    mlp_dwconv=mlp_dwconv,
                    before_attn_dwconv=before_attn_dwconv,
                    pre_norm=pre_norm,
                    ffn_expansion_factor=ffn_expansion_factor,
                    patch_size=patch_size
                    ) for j in range(depth[0])],
        )
        self.stages.append(stage)
        # cur += depth[1]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1) #平均池化

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(1):
            x = self.dimensreduct_layers[i](x) # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.forward_features(x)
        x = self.avgpool(x)  #
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    
    # for IP dataset in_chans=200 num_classes=16 patchsize=11
    model = GLMGT(in_chans=200, num_classes=16, patch_size=11)

    print(model)
    x = torch.randn(100,1,200,11,11)
    y = model(x)
    print(y.shape)
