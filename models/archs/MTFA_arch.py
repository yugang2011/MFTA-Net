import math
import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from einops import rearrange
import functools
import torch.nn.functional as F
from models.vgg_model import VGGFeatureExtractor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)





class CAB(nn.Module):
    """
    Channel attention bolck.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16,in here set 30
    """
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    """
    MLP is aim to Learn adaptive parameters
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
## ........##

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, dim,
                input_resolution,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim,dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv) # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x


class AttenBlocks(nn.Module):
    """ A series of attention blocks for one RHAG.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # OCAB
        self.overlap_attn = OCAB(
                            dim=dim,
                            input_resolution=input_resolution,
                            window_size=window_size,
                            overlap_ratio=overlap_ratio,
                            num_heads=num_heads,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            mlp_ratio=mlp_ratio,
                            norm_layer=norm_layer
                            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])

        x = self.overlap_attn(x, x_size, params['rpi_oca'])

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RHAG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class ResidualBlock(nn.Module):
    """
    Base blocks aim to Accelerate convergence

    """
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)




class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        """
        Deep convolutional layers reduce computational complexity and are used here to capture high-frequency information
        """
        self.dconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=ksize, stride=stride, groups=in_channels,padding=1)
        self.pconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1)
        self.act = nn.SiLU() if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        x = self.act(x)
        return x


class HFEM(nn.Module):
    """
    High frequency reconstruction module for enhancing network capture.
    """
    def __init__(self, nf):
        super(HFEM, self).__init__()

        # LFE
        self.conv_l = nn.Conv2d(nf//2, nf//2, 3, 1, 1)

        # HFE
        self.conv_dw = DWConv(in_channels=nf//2,out_channels=nf//2,ksize=3)
        self.conv_h = nn.Conv2d(nf//2, nf//2, 1, 1)

        # final
        self.conv_f = nn.Conv2d(nf, nf, 1, 1)

        self.act = nn.GELU()

    def forward(self, x):
        x_l = x[:, 0:32, :, :]
        x_h = x[:, 32:64, :, :]

        x_l = self.act(self.conv_l(x_l))
        x_h = self.act(self.conv_h(self.conv_dw(x_h)))

        out = self.conv_f(torch.cat([x_l,x_h],dim=1))

        return out


@ARCH_REGISTRY.register()
class MTFA(nn.Module):
    r""" .
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=1,
                 embed_dim=96,

                 depths=(4, 4, 4, ),
                 num_heads=(4, 4, 4),
                 window_size=16,
                 compress_ratio=24,
                 squeeze_factor=24,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 img_range=1.,
                 resi_connection='1conv',
                 **kwargs):
        super(MTFA, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_feat = 64
        # tranformer
        n_blks = [4, 4, 4]
        self.mean = torch.zeros(1, 1, 1, 1)
        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # ------------------------- Initial feature extraction -------------------------------- #
        self.conv_init = nn.Conv2d(in_chans, 3, 3, 1, 1)
        # ------------------------- Pre trained VGG extraction branch ------------------------- #
        layer_weights_L1 = {'conv1_2': 1.}
        self.vgg_L1 = VGGFeatureExtractor(
            layer_name_list=list(layer_weights_L1.keys()),
            vgg_type='vgg19',
            use_input_norm=True)
        self.conv_vgg_L1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)

        layer_weights_L2 = {'conv2_2': 1.}
        self.vgg_L2 = VGGFeatureExtractor(
            layer_name_list=list(layer_weights_L2.keys()),
            vgg_type='vgg19',
            use_input_norm=True)
        self.conv_vgg_L2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        layer_weights_L3 = {'conv3_4': 1.}
        self.vgg_L3 = VGGFeatureExtractor(
            layer_name_list=list(layer_weights_L3.keys()),
            vgg_type='vgg19',
            use_input_norm=True)
        self.conv_vgg_L3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        # ---- Convolutional residual blocks before transformers, used to stabilize the training process ----------
        block = functools.partial(ResidualBlock, nf=num_feat)

        self.conv_L1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[0])
        self.conv_before_upsample_L1 = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))

        self.conv_L2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])
        self.conv_before_upsample_L2 = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))

        self.conv_L3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[2])

        self.act = nn.ReLU(inplace=True)

        # ----Using transformers for deep feature extraction to model the connections between multiple tasks -- #
        self.conv_first = nn.Conv2d(128, embed_dim, 3, 1, 1)
        self.conv_second = nn.Conv2d(128, embed_dim, 3, 1, 1)
        self.conv_third = nn.Conv2d(128, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Hybrid Attention Groups (RHAG)
        self.layer_L1 = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
        self.norm = norm_layer(self.num_features)
        self.conv_after_body_L1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.layer_L2 = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
        self.conv_after_body_L2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.layer_L3 = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
        self.conv_after_body_L3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        # ------------------------- high quality image reconstruction ------------------------- #
        # SR branch
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        block = functools.partial(ResidualBlock, nf=64)
        # -------------------------------------------------------------------------
        self.sam_sr_L3 = SAM(nf=64)
        self.res_block_sr_L3 = make_layer(block, n_layers=4)
        self.HFEM_sr_L3 = HFEM(nf=num_feat)
        self.conv_out_sr_L3 = nn.Conv2d(64, 64, 3, 1, 1)
        # -------------------------------------------------------------------------
        self.conv_in_sr_L2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.sam_sr_L2 = SAM(nf=64)
        self.res_block_sr_L2 = make_layer(block, n_layers=8)
        self.HFEM_sr_L2 = HFEM(nf=num_feat)
        self.conv_out_sr_L2 = nn.Conv2d(64, 64, 3, 1, 1)
        # -------------------------------------------------------------------------
        self.conv_in_sr_L1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.sam_sr_L1 = SAM(nf=64)
        self.res_block_sr_L1 = make_layer(block, n_layers=12)
        self.HFEM_sr_L1 = HFEM(nf=num_feat)
        self.conv_out_sr_L1 = nn.Conv2d(64, 64, 3, 1, 1)
        # -------------------------------------------------------------------------
        self.conv_out_sr = nn.Conv2d(64, 1, 3, 1, 1)

        # SY branch
        self.conv_before_upsample_ = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        # -------------------------------------------------------------------------
        self.sam_sy_L3 = SAM(nf=64)
        self.res_block_sy_L3 = make_layer(block, n_layers=4)
        self.HFEM_sy_L3 = HFEM(nf=num_feat)
        self.conv_out_sy_L3 = nn.Conv2d(64, 64, 3, 1, 1)
        # -------------------------------------------------------------------------
        self.conv_in_sy_L2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.sam_sy_L2 = SAM(nf=64)
        self.res_block_sy_L2 = make_layer(block, n_layers=8)
        self.HFEM_sy_L2 = HFEM(nf=num_feat)
        self.conv_out_sy_L2 = nn.Conv2d(64, 64, 3, 1, 1)
        # -------------------------------------------------------------------------
        self.conv_in_sy_L1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.sam_sy_L1 = SAM(nf=64)
        self.res_block_sy_L1 = make_layer(block, n_layers=12)    # 16 12 8
        self.HFEM_sy_L1 = HFEM(nf=num_feat)
        self.conv_out_sy_L1 = nn.Conv2d(64, 64, 3, 1, 1)
        # -------------------------------------------------------------------------
        self.conv_out_sy = nn.Conv2d(64, 1, 3, 1, 1)

        # DS
        self.out_sr_L3 = nn.Conv2d(64, 1, 3, 1, 1)
        self.out_sr_L2 = nn.Conv2d(64, 1, 3, 1, 1)
        self.out_sy_L3 = nn.Conv2d(64, 1, 3, 1, 1)
        self.out_sy_L2 = nn.Conv2d(64, 1, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, tar):
        # VGG19
        fea_vgg_L1 = self.act(self.conv_init(tar))
        fea_vgg_L1 = self.vgg_L1(fea_vgg_L1)

        fea_vgg_L1 = self.conv_vgg_L1(fea_vgg_L1['conv1_2'])

        fea_vgg_L2 = self.act(self.conv_init(tar))
        fea_vgg_L2 = self.vgg_L2(fea_vgg_L2)
        fea_vgg_L2 = self.conv_vgg_L2(fea_vgg_L2['conv2_2'])

        fea_vgg_L3 = self.act(self.conv_init(tar))
        fea_vgg_L3 = self.vgg_L3(fea_vgg_L3)
        fea_vgg_L3 = self.conv_vgg_L3(fea_vgg_L3['conv3_4'])

        fea_L1 = self.act(self.conv_L1(tar))  # [B,N,160,160]
        fea_L1 = self.blk_L1(fea_L1)
        fea_L1 = self.conv_first(torch.cat([fea_L1, fea_vgg_L1], dim=1))

        x_size_L1 = (fea_L1.shape[2], fea_L1.shape[3])
        attn_mask_L1 = self.calculate_mask(x_size_L1).to(fea_L1.device)
        params_L1 = {'attn_mask': attn_mask_L1, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}
        fea_L1 = self.patch_embed(fea_L1)
        fea_L1 = self.pos_drop(fea_L1)

        fea_L1 = self.layer_L1(fea_L1, x_size_L1, params_L1)
        fea_L1 = self.norm(fea_L1)
        fea_L1 = self.patch_unembed(fea_L1, x_size_L1)
        fea_L1 = self.conv_after_body_L1(fea_L1)+fea_L1
        fea_L1 = self.conv_before_upsample_L1(fea_L1)

        fea_L2 = self.act(self.conv_L2(fea_L1))  # [B,N,80,80]
        fea_L2 = self.blk_L2(fea_L2)
        fea_L2 = self.conv_second(torch.cat([fea_L2, fea_vgg_L2], dim=1))
        x_size_L2 = (fea_L2.shape[2], fea_L2.shape[3])

        attn_mask_L2 = self.calculate_mask(x_size_L2).to(fea_L2.device)
        params_L2 = {'attn_mask': attn_mask_L2, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}
        fea_L2 = self.patch_embed(fea_L2)
        fea_L2 = self.pos_drop(fea_L2)

        fea_L2 = self.layer_L2(fea_L2, x_size_L2, params_L2)
        fea_L2 = self.norm(fea_L2)
        fea_L2 = self.patch_unembed(fea_L2, x_size_L2)
        fea_L2 = self.conv_after_body_L2(fea_L2)+fea_L2
        fea_L2 = self.conv_before_upsample_L2(fea_L2)

        fea_L3 = self.act(self.conv_L3(fea_L2))  # [B,N,40,40]
        fea_L3 = self.blk_L3(fea_L3)
        fea_L3 = self.conv_third(torch.cat([fea_L3, fea_vgg_L3], dim=1))
        x_size_L3 = (fea_L3.shape[2], fea_L3.shape[3])
        attn_mask_L3 = self.calculate_mask(x_size_L3).to(fea_L3.device)
        params_L3 = {'attn_mask': attn_mask_L3, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}

        fea_L3 = self.patch_embed(fea_L3)
        fea_L3 = self.pos_drop(fea_L3)

        fea_L3 = self.layer_L3(fea_L3, x_size_L3, params_L3)
        fea_L3 = self.norm(fea_L3)
        fea_L3 = self.patch_unembed(fea_L3, x_size_L3)
        fea_L3 = self.conv_after_body_L3(fea_L3)+fea_L3

        # MR image Super-resolution
        x_sr_L3 = self.conv_before_upsample(fea_L3)
        x_sr_L3 = self.sam_sr_L3(x_sr_L3, x_sr_L3)
        x_sr_L3_ = self.res_block_sr_L3(x_sr_L3)+x_sr_L3
        x_sr_L3_ = self.HFEM_sr_L3(x_sr_L3_)
        x_sr_up_L3 = F.interpolate(x_sr_L3_, scale_factor=2, mode='bilinear', align_corners=False)
        x_sr_up_L3 = self.conv_out_sr_L3(x_sr_up_L3)

        x_sr_L2_ = self.sam_sr_L2(self.conv_in_sr_L2(torch.cat([x_sr_up_L3, fea_L2], dim=1)), x_sr_up_L3)
        x_sr_L2_ = self.res_block_sr_L2(x_sr_L2_)+x_sr_up_L3
        x_sr_L2_ = self.HFEM_sr_L2(x_sr_L2_)
        x_sr_up_L2 = F.interpolate(x_sr_L2_, scale_factor=2, mode='bilinear', align_corners=False)
        x_sr_up_L2 = self.conv_out_sr_L2(x_sr_up_L2)

        x_sr_L1_ = self.sam_sr_L1(self.conv_in_sr_L1(torch.cat([x_sr_up_L2, fea_L1], dim=1)), x_sr_up_L2)
        x_sr_L1_ = self.res_block_sr_L1(x_sr_L1_)+x_sr_up_L2
        x_sr_L1_ = self.HFEM_sr_L1(x_sr_L1_)
        x_sr_up_L1 = self.conv_out_sr_L1(x_sr_L1_)
        x_sr_final = self.conv_out_sr(x_sr_up_L1)

        # MR image Synthesis
        x_sy_L3 = self.conv_before_upsample_(fea_L3)
        x_sy_L3 = self.sam_sy_L3(x_sy_L3, x_sy_L3)
        x_sy_L3_ = self.res_block_sy_L3(x_sy_L3) + x_sy_L3
        x_sy_L3_ = self.HFEM_sy_L3(x_sy_L3_)
        x_sy_up_L3 = F.interpolate(x_sy_L3_, scale_factor=2, mode='bilinear', align_corners=False)
        x_sy_up_L3 = self.conv_out_sy_L3(x_sy_up_L3)

        x_sr_L2_ = self.sam_sy_L2(self.conv_in_sy_L2(torch.cat([x_sy_up_L3, fea_L2], dim=1)), x_sy_up_L3)
        x_sy_L2_ = self.res_block_sy_L2(x_sr_L2_) + x_sy_up_L3
        x_sy_L2_ = self.HFEM_sy_L2(x_sy_L2_)
        x_sy_up_L2 = F.interpolate(x_sy_L2_, scale_factor=2, mode='bilinear', align_corners=False)
        x_sy_up_L2 = self.conv_out_sy_L2(x_sy_up_L2)

        x_sr_L1_ = self.sam_sy_L1(self.conv_in_sy_L1(torch.cat([x_sy_up_L2, fea_L1], dim=1)), x_sy_up_L2)
        x_sy_L1_ = self.res_block_sy_L1(x_sr_L1_) + x_sy_up_L2
        x_sy_L1_ = self.HFEM_sy_L1(x_sy_L1_)
        x_sy_up_L1 = self.conv_out_sy_L1(x_sy_L1_)
        x_sy_final = self.conv_out_sy(x_sy_up_L1)
        x_sr_final_L3 = x_sr_final_L2=x_sy_final_L3=x_sy_final_L2 = 0

        if self.training:
            x_sr_final_L3 = self.out_sr_L3(x_sr_L3_)
            x_sr_final_L2 = self.out_sr_L2(x_sr_L2_)
            x_sy_final_L3 = self.out_sy_L3(x_sy_L3_)
            x_sy_final_L2 = self.out_sy_L2(x_sy_L2_)

        # 0,1,2,3,4,5
        return [x_sr_final, x_sy_final, x_sr_final_L3, x_sr_final_L2,  x_sy_final_L3, x_sy_final_L2]
