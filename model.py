import torch
from torch import nn
import torch.nn.functional as F
import settings
from timm.models.layers import DropPath, to_2tuple
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    ZeroPad2d = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    img = ZeroPad2d(input_data)
    col = torch.zeros([N, C, filter_h,filter_w, out_h,out_w]).cuda()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride,]

    col = col.reshape(N, C, filter_h*filter_w, out_h*out_w)

    return col


def col2im(col, orisize, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = orisize
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, C, filter_h, filter_w,out_h, out_w)
    img = torch.zeros([N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1]).cuda()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class NLWT(nn.Module):
    def __init__(self):
        super(NLWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return NLWT_init(x)

def NLWT_init(x):
    U1ImulP1I = np.array([[0.9124,   -0.0256,    0.4083,    0.0115],
                          [0.0256,    0.9124,    0.0115,   -0.4083],
                          [-0.4083,   0.0115,    0.9124,    0.0256],
                          [0.0115,    0.4083,   -0.0256,    0.9124]])
    U2ImulP2Imul4 = np.array([[1.3066,    0.5412,    0.5412,    1.3066],
                              [-0.5412,   1.3066,    1.3066,   -0.5412],
                              [-0.4671,   -1.3349,    1.3349,    0.4671],
                              [1.3349,    -0.4671,    0.4671,   -1.3349]])
    U1ImulP1I = torch.cuda.FloatTensor(U1ImulP1I).unsqueeze(0).unsqueeze(0)
    U2ImulP2Imul4 = torch.cuda.FloatTensor(U2ImulP2Imul4).unsqueeze(0).unsqueeze(0)

    b, c, h, w = x.size()
    orisize = x.size()

    xT_col = im2col(x, 2, 2, stride=2, pad=0);
    x1 = U1ImulP1I @ xT_col;

    h1 = h // 2
    w1 = w // 2
    T2 = x1[:, :, 1, :].reshape(b, c, h1, w1);
    T3 = x1[:, :, 2, :].reshape(b, c, h1, w1);
    T4 = x1[:, :, 3, :].reshape(b, c, h1, w1);

    T22 = torch.roll(T2, shifts=-1, dims=2)
    T32 = torch.roll(T3, shifts=-1, dims=3)
    T42 = torch.roll(T4, shifts=(-1, -1), dims=(2, 3))

    x1[:, :, 1, :] = T22.flatten(2);
    x1[:, :, 2, :] = T32.flatten(2);
    x1[:, :, 3, :] = T42.flatten(2);

    x2 = U2ImulP2Imul4 @ x1;

    A_low0 = x2[:, :, 0, :].reshape(b, c, h1, w1);
    B_high1 = x2[:, :, 1, :].reshape(b, c, h1, w1);
    C_high2 = x2[:, :, 2, :].reshape(b, c, h1, w1);
    D_high3 = x2[:, :, 3, :].reshape(b, c, h1, w1);

    return A_low0, B_high1, C_high2, D_high3, orisize

class INLWT(nn.Module):
    def __init__(self):
        super(INLWT, self).__init__()
        self.requires_grad = False

    def forward(self, A_low0, B_high1, C_high2, D_high3,orisize):
        return INLWT_init(A_low0, B_high1, C_high2, D_high3, orisize)

def INLWT_init(A_low0,B_high1,C_high2,D_high3,orisize):
    P2mulU2 = np.array([[1.3066,   -0.5412,   -0.4671,    1.3349],
                        [0.5412,    1.3066,   -1.3349,   -0.4671],
                        [0.5412,    1.3066,    1.3349,    0.4671],
                        [1.3066,   -0.5412,    0.4671,   -1.3349]])
    P1mulU1div4 = np.array([[0.2281,    0.0064,   -0.1021,    0.0029],
                            [-0.0064,   0.2281,    0.0029,    0.1021],
                            [0.1021,    0.0029,    0.2281,   -0.0064],
                            [0.0029,   -0.1021,    0.0064,    0.2281]])
    P2mulU2 = torch.cuda.FloatTensor(P2mulU2).unsqueeze(0).unsqueeze(0)
    P1mulU1div4 = torch.cuda.FloatTensor(P1mulU1div4).unsqueeze(0).unsqueeze(0)

    b, c, h1, w1 = A_low0.size()
    A = A_low0.reshape(b, c, 1, h1 * w1);
    B = B_high1.reshape(b, c, 1, h1 * w1);
    C = C_high2.reshape(b, c, 1, h1 * w1);
    D = D_high3.reshape(b, c, 1, h1 * w1);

    Y1 = torch.cat([A, B, C, D], dim=2)
    Y2 = P2mulU2 @ Y1;
    t2 = Y2[:, :, 1, :].reshape(b, c, h1, w1);
    t3 = Y2[:, :, 2, :].reshape(b, c, h1, w1);
    t4 = Y2[:, :, 3, :].reshape(b, c, h1, w1);

    t22 = torch.roll(t2, shifts=1, dims=2)
    t32 = torch.roll(t3, shifts=1, dims=3)
    t42 = torch.roll(t4, shifts=(1, 1), dims=(2, 3))

    Y2[:, :, 1, :] = t22.flatten(2)
    Y2[:, :, 2, :] = t32.flatten(2)
    Y2[:, :, 3, :] = t42.flatten(2)

    Y3 = P1mulU1div4 @ Y2;
    rst = col2im(Y3, orisize, 2, 2, stride=2, pad=0);

    return rst


class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Sequential(nn.Conv2d(inputchannel, outchannel, kernel_size, stride), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.conv(self.padding(x))
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qk = nn.Linear(dim, dim * 2, bias=False)
        self.v = nn.Linear(dim, dim * 1, bias=False)
        self.cat_v = nn.Linear(dim*2, dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.k_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
            self.v_mix_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None,cat_v=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        if self.q_bias is not None:
            qk_bias = torch.cat((self.q_bias, self.k_bias))
        qk = F.linear(input=x, weight=self.qk.weight, bias=qk_bias)
        v = F.linear(input=x, weight=self.v.weight, bias=self.v_bias)
        qk = qk.reshape(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v = torch.cat([v,cat_v],dim=2)
        v = F.linear(input=v, weight=self.cat_v.weight, bias=self.v_mix_bias).reshape(B_, N, self.num_heads, -1).permute( 0, 2, 1, 3)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# class Detail_enhanced_Attention_Feed_forward_Layer(nn.Module):
#     def __init__(self, dim, mlp_ratio=4, drop=0., norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.MLP_DCL = MLP_DCL(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
#         self.DA = Dual_branch_Attention(dim)
#
#     def forward(self, x,H,W):
#         out = x + self.drop_path(self.norm2(self.DA(self.MLP_DCL(x,H,W),H,W)))
#         return out

class MLP_DCL(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.DCL = Depth_wise_Convolutional_Layer(hidden_features)

    def forward(self, x,H,W):
        x = self.fc1(x)
        x = self.act(x + self.DCL(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x_mlp = self.drop(x)

        return x_mlp

class Depth_wise_Convolutional_Layer(nn.Module):
    def __init__(self, dim,r=4):
        super(Depth_wise_Convolutional_Layer, self).__init__()
        self.hide_channel = dim//r
        self.dwconv = nn.Sequential(nn.Conv2d(dim,self.hide_channel,1,1),nn.Conv2d(self.hide_channel, self.hide_channel, 3, 1, 1, bias=True, groups=self.hide_channel),nn.Conv2d(self.hide_channel,dim,1,1))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class NAFBlock_pre(nn.Module):
    def __init__(self, dim, DW_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = dim * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        self.norm1 = LayerNorm2d(dim)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)


    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = x * self.beta

        return y

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class Dual_branch_Attention(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.fuse1 = nn.Linear(3*in_channels, in_channels)
        self.fuse2 = nn.Linear(3*in_channels, reduce_channels)
        self.norm = nn.LayerNorm(in_channels)
        # self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()
        self.NLWT = NLWT()

    def forward(self, x,H,W):
        B, L, C = x.shape
        F = self.norm(x)
        F_= F
        F = F.transpose(1,2).reshape(B,C,H,W)
        FL0, FH1, FH2, FH3, orisize = self.NLWT(F)
        FH123 = self.fuse1(torch.cat([FH1.flatten(2).transpose(1,2), FH2.flatten(2).transpose(1,2), FH3.flatten(2).transpose(1,2)],dim=-1))
        FL0_pool = FL0.flatten(2).transpose(1,2).mean(1, keepdim=True)
        FH123_pool = FH123.mean(1, keepdim=True)
        F_pool = F_.mean(1, keepdim=True)

        F_global = self.act_fn(self.fuse2(torch.cat([FL0_pool, FH123_pool, F_pool], dim=-1)))
        F_local = self.act_fn(self.local_reduce(F_))

        F_CA = self.gate_fn(self.channel_select(F_global)) # [B, 1, C]
        F_SA = self.gate_fn(self.spatial_select(torch.cat([F_local, F_global.expand(-1, L, -1)], dim=-1))) # [B, L, 1]

        attn = F_CA * F_SA  # [B, L, C]
        out = x * attn
        return out

class CTHAB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.dimdiv2 = dim//2
        self.norm1 = norm_layer(self.dimdiv2)
        self.attn = WindowAttention(
            self.dimdiv2, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.NAF_pre = NAFBlock_pre(self.dimdiv2)
        self.fuse1 = nn.Sequential(nn.Conv2d(dim, dim, 1, 1), nn.LeakyReLU(0.2))
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.MLP_DCL = MLP_DCL(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.DA = Dual_branch_Attention(dim)
        # self.DAFL = Detail_enhanced_Attention_Feed_forward_Layer(dim,mlp_ratio,drop,norm_layer)

    def forward(self, x, H, W):
        x1 = None
        x2 = None
        if len(x.shape) == 3:
            B, L, C = x.shape
            Cdiv2 = C // 2
            shortcut = x
            x1, x2 = x.chunk(2, dim=2)
            x1 = x1.view(B, H, W, Cdiv2).contiguous()
            x2 = x2.transpose(1, 2).contiguous().view(B, Cdiv2, H, W)
        elif len(x.shape) == 4:
            B, H, W, C = x.shape
            Cdiv2 = C // 2
            shortcut = x.view(B, H * W, C).contiguous()
            x1, x2 = x.chunk(2, dim=3)
            x2 = x2.permute(0, 3, 1, 2).contiguous()

        x2_conv = self.NAF_pre(x2).flatten(2).transpose(1,2)
        attn_mask = None
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.window_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.window_size * self.window_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = shift_attn_mask

        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x1 = x1
        # partition windows
        x1_windows = window_partition(shifted_x1, self.window_size)  # nW*B, window_size, window_size, C
        x1_windows = x1_windows.view(-1, self.window_size * self.window_size, Cdiv2)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn1_windows = self.attn(x1_windows, mask=attn_mask,cat_v=x2_conv.reshape(-1, self.window_size * self.window_size, Cdiv2)) # nW*B, window_size*window_size, C

        # merge windows
        attn1_windows = attn1_windows.view(-1, self.window_size, self.window_size, Cdiv2)
        shifted_x1 = window_reverse(attn1_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x1
        x1 = x1.view(B, H * W, Cdiv2)

        x1_MSA = self.drop_path(self.norm1(x1)).transpose(1, 2).reshape(B, Cdiv2, H, W)
        x = self.fuse1(torch.cat([x1_MSA, x2_conv.transpose(1,2).reshape(B, Cdiv2, H, W)], dim=1)).flatten(2).transpose(1, 2) + shortcut

        # DAFL
        out = x + self.drop_path(self.norm2(self.DA(self.MLP_DCL(x,H,W),H,W)))

        return out

class Multi_level_Wavelet_Network_Based_on_CNN_Transformer_Hybrid_Attention(nn.Module):
    def __init__(self, in_channel=3, channel=settings.channel):
        super().__init__()
        self.input_resolution = settings.patch_size
        self.window_size = settings.window_size
        self.depth = settings.depth
        self.heads = settings.heads
        self.convert = nn.Sequential(nn.Conv2d(in_channel, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.high_fuse1 = convd(channel * 3, channel, 3, 1)
        self.high_fuse2 = convd(channel * 3, channel, 3, 1)
        self.high_fuse3 = convd(channel * 3, channel, 3, 1)
        self.high_fuse4 = convd(channel * 3, channel, 3, 1)
        self.high_ifuse1 = convd(channel, channel*3, 3, 1)
        self.high_ifuse2 = convd(channel, channel*3, 3, 1)
        self.high_ifuse3 = convd(channel, channel*3, 3, 1)
        self.high_ifuse4 = convd(channel, channel*3, 3, 1)
        self.concat_fuse1 = convd(channel * 2, channel, 3, 1)
        self.concat_fuse2 = convd(channel * 2, channel, 3, 1)
        self.concat_fuse3 = convd(channel * 2, channel, 3, 1)
        self.concat_fuse4 = convd(channel * 2, channel, 3, 1)
        self.out = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(channel, 3, 1, 1))
        self.drop_path_rate = 0.1
        per1 = self.depth
        per2 = self.depth
        self.num_enc_layers = [per1, per1, per1, per1, per1, per1, per1, per1, per1]
        self.enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.num_enc_layers))]
        self.num_dec_layers = [per2, per2, per2, per2]
        self.dec_dpr_inv = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.num_dec_layers))]
        self.dec_dpr = self.dec_dpr_inv[::-1]
        self.De_Lev0_CTHAB = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution,self.input_resolution],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[0*per1 + i])
            for i in range(self.depth)])
        self.De_Lev1_CTHAB1 = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//2,self.input_resolution//2],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[1*per1 + i])
            for i in range(self.depth)])
        self.De_Lev1_CTHAB2 = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//2,self.input_resolution//2],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[2*per1 + i])
            for i in range(self.depth)])
        self.De_Lev2_CTHAB1 = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//4,self.input_resolution//4],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[3*per1 + i])
            for i in range(self.depth)])
        self.De_Lev2_CTHAB2 = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//4,self.input_resolution//4],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[4*per1 + i])
            for i in range(self.depth)])
        self.De_Lev3_CTHAB1 = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//8,self.input_resolution//8],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[5*per1 + i])
            for i in range(self.depth)])
        self.De_Lev3_CTHAB2 = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//8,self.input_resolution//8],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[6*per1 + i])
            for i in range(self.depth)])
        self.De_Lev4_CTHAB1 = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//16,self.input_resolution//16],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[7*per1 + i])
            for i in range(self.depth)])
        self.De_Lev4_CTHAB2 = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//16,self.input_resolution//16],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[8*per1 + i])
            for i in range(self.depth)])
        self.Re_Lev3_CTHAB = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//8,self.input_resolution//8],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[0*per2 + i])
            for i in range(self.depth)])
        self.Re_Lev2_CTHAB = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//4,self.input_resolution//4],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[1*per2 + i])
            for i in range(self.depth)])
        self.Re_Lev1_CTHAB = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution//2,self.input_resolution//2],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[2*per2 + i])
            for i in range(self.depth)])
        self.Re_Lev0_CTHAB = nn.ModuleList([
            CTHAB(dim=channel,input_resolution=[self.input_resolution,self.input_resolution],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[3*per2 + i])
            for i in range(self.depth)])
        self.NLWT = NLWT()
        self.INLWT = INLWT()

    def check_image_size(self, x):
        _, _, h, w = x.size()
        size = 128
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def restore_image_size(self, x, ori_size):
        return x[:, :, :ori_size[0], :ori_size[1]]

    def split(self, x):
        channel = x.shape[1] // 3
        A = x[:, 0:channel, :, :]
        B = x[:, channel:channel * 2, :, :]
        C = x[:, channel * 2:channel * 3, :, :]
        return A, B, C

    def forward(self, x):
        x_check, ori_size = self.check_image_size(x)
        DL0 = self.convert(x_check)
        b,c0,h0,w0 = DL0.shape
        DL0 = DL0.flatten(2).transpose(1, 2).contiguous()
        for blk in self.De_Lev0_CTHAB:
            DL0 = blk(DL0,h0,w0)
        DL0 = DL0.transpose(1,2).view(b,c0,h0,w0)

        DL1, DH1_1, DH1_2, DH1_3, DL0_orisize = self.NLWT(DL0)
        b,c1,h1,w1=DL1.shape
        DH1_123 = self.high_fuse1(torch.cat([DH1_1, DH1_2, DH1_3],dim=1))
        DL1 = DL1.flatten(2).transpose(1, 2).contiguous()
        DH1_123= DH1_123.flatten(2).transpose(1, 2).contiguous()
        for blk in self.De_Lev1_CTHAB1:
            DL1 = blk(DL1,h1,w1)
        for blk in self.De_Lev1_CTHAB2:
            DH1_123 = blk(DH1_123, h1, w1)
        DL1 = DL1.transpose(1,2).view(b,c1,h1,w1)
        RH1_123 = DH1_123.transpose(1,2).view(b,c1,h1,w1)
        RH1_1, RH1_2, RH1_3 = self.split(self.high_ifuse1(RH1_123))

        DL2, DH2_1, DH2_2, DH2_3, DL1_orisize = self.NLWT(DL1)
        b,c2,h2,w2 = DL2.shape
        DH2_123 = self.high_fuse2(torch.cat([DH2_1, DH2_2, DH2_3],dim=1))
        DL2 = DL2.flatten(2).transpose(1, 2).contiguous()
        DH2_123= DH2_123.flatten(2).transpose(1, 2).contiguous()
        for blk in self.De_Lev2_CTHAB1:
            DL2 = blk(DL2,h2,w2)
        for blk in self.De_Lev2_CTHAB2:
            DH2_123 = blk(DH2_123,h2,w2)
        DL2 = DL2.transpose(1,2).view(b,c2,h2,w2)
        RH2_123 = DH2_123.transpose(1,2).view(b,c2,h2,w2)
        RH2_1, RH2_2, RH2_3 = self.split(self.high_ifuse2(RH2_123))

        DL3, DH3_1, DH3_2, DH3_3, DL2_orisize = self.NLWT(DL2)
        b,c3,h3,w3 = DL3.shape
        DH3_123 = self.high_fuse3(torch.cat([DH3_1, DH3_2, DH3_3],dim=1))
        DL3 = DL3.flatten(2).transpose(1, 2).contiguous()
        DH3_123= DH3_123.flatten(2).transpose(1, 2).contiguous()
        for blk in self.De_Lev3_CTHAB1:
            DL3 = blk(DL3,h3,w3)
        for blk in self.De_Lev3_CTHAB2:
            DH3_123 = blk(DH3_123,h3,w3)
        DL3 = DL3.transpose(1,2).view(b,c2,h3,w3)
        RH3_123 = DH3_123.transpose(1,2).view(b,c2,h3,w3)
        RH3_1, RH3_2, RH3_3 = self.split(self.high_ifuse3(RH3_123))

        DL4, DH4_1, DH4_2, DH4_3, DL3_orisize = self.NLWT(DL3)
        b,c4,h4,w4 = DL4.shape
        DH4_123 = self.high_fuse4(torch.cat([DH4_1, DH4_2, DH4_3],dim=1))
        DL4 = DL4.flatten(2).transpose(1, 2).contiguous()
        DH4_123= DH4_123.flatten(2).transpose(1, 2).contiguous()
        for blk in self.De_Lev4_CTHAB1:
            DL4 = blk(DL4,h4,w4)
        for blk in self.De_Lev4_CTHAB2:
            DH4_123 = blk(DH4_123,h4,w4)
        RL4 = DL4.transpose(1,2).view(b,c2,h4,w4)
        RH4_123 = DH4_123.transpose(1,2).view(b,c2,h4,w4)
        RH4_1, RH4_2, RH4_3 = self.split(self.high_ifuse4(RH4_123))

        RL3 = self.INLWT(RL4, RH4_1, RH4_2, RH4_3, DL3_orisize)
        RL3_fuse = self.concat_fuse1(torch.cat([RL3, DL3], dim=1))
        RL3_fuse = RL3_fuse.flatten(2).transpose(1, 2).contiguous()
        for blk in self.Re_Lev3_CTHAB:
            RL3_fuse = blk(RL3_fuse,h3,w3)
        RL3_fuse = RL3_fuse.transpose(1,2).view(b,c3,h3,w3)

        RL2 = self.INLWT(RL3_fuse, RH3_1, RH3_2, RH3_3, DL2_orisize)
        RL2_fuse = self.concat_fuse2(torch.cat([RL2, DL2], dim=1))
        RL2_fuse = RL2_fuse.flatten(2).transpose(1, 2).contiguous()
        for blk in self.Re_Lev2_CTHAB:
            RL2_fuse = blk(RL2_fuse,h2,w2)
        RL2_fuse = RL2_fuse.transpose(1,2).view(b,c2,h2,w2)

        RL1 = self.INLWT(RL2_fuse, RH2_1, RH2_2, RH2_3, DL1_orisize)
        RL1_fuse = self.concat_fuse3(torch.cat([RL1, DL1], dim=1))
        RL1_fuse = RL1_fuse.flatten(2).transpose(1, 2).contiguous()
        for blk in self.Re_Lev1_CTHAB:
            RL1_fuse = blk(RL1_fuse,h1,w1)
        RL1_fuse = RL1_fuse.transpose(1,2).view(b,c1,h1,w1)

        RL0 = self.INLWT(RL1_fuse, RH1_1, RH1_2, RH1_3, DL0_orisize)
        RL0_fuse = self.concat_fuse4(torch.cat([RL0, DL0], dim=1))
        RL0_fuse = RL0_fuse.flatten(2).transpose(1, 2).contiguous()
        for blk in self.Re_Lev0_CTHAB:
            RL0_fuse = blk(RL0_fuse,h0,w0)
        RL0_fuse = RL0_fuse.transpose(1,2).view(b,c0,h0,w0)

        y = self.restore_image_size(self.out(RL0_fuse), ori_size)
        out = x + y
        return out


