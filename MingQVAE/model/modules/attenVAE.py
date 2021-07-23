import torch
import torch.nn as nn
import torch.nn.functional as F

# swish激活函数
class nonlinearity(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = x * torch.sigmoid(x)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        return out

# 标准化采用GroupNorm
def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# 上采样块
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv=False):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest") # 2倍最近邻上采样
        if self.with_conv:
            x = self.conv(x)
        return x

# 下采样块
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv=False):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0,1,0,1), mode="constant", value=0) # 图像右边和下边各填充1个单位，左上不变
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.2, temb_channels=512):
        super().__init__()

        self.in_channels = in_channels # 输入深度
        out_channels = in_channels if out_channels is None else out_channels # 当输出深度未指定时，默认与输入深度相同
        self.out_channels = out_channels # 输出深度

        self.conv1 = nn.Sequential(
            Normalize(in_channels),
            nonlinearity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) # 卷积核(3,3,in)，步长1，扩充1（保持图像尺寸不变）
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.conv2 = nn.Sequential(
            Normalize(out_channels),
            nonlinearity(dropout=dropout), # 带一个后dropout层
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )    

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) # 卷积核(1,1),相当于直接连接

    def forward(self, x, temb=None):
        h = x
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x+h # 跨层连接

# 注意力块，全局化特征
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)
        return x+h_

# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channels, C1_channels, z_channels, resolution, layer_res_blocks_num, attn_resolutions, ch_mult=(1,2,4,8), double_z=True, dropout=False, **ignore_kwargs):
        super().__init__()
        self.C = in_channels
        self.C1 = C1_channels
        self.nz = 2*z_channels if double_z else z_channels
        self.m = len(ch_mult) # 下采样层数
        self.layer_res_blocks_num = layer_res_blocks_num # 每层下采样模块中含的串行残差块数量
        self.temp_ch = 0 # 残差块额外信息维度
        
        # 输入卷积层
        self.conv_in = nn.Conv2d(in_channels, C1_channels, kernel_size=3, stride=1, padding=1)

        # 下采样模块（m-1倍）
        curr_res = resolution # 记录实时的图像分辨率
        block_in = C1_channels # 初始输入深度
        
        self.downsamples = nn.Sequential()
        for i in range(self.m): # 逐层添加残差块
            block_out = C1_channels*ch_mult[i] # 每层的输出深度
            for j in range(self.layer_res_blocks_num):
                self.downsamples.add_module('res_block_l{}b{}'.format(i,j), ResidualBlock(in_channels=block_in, out_channels=block_out, dropout=dropout, temb_channels=self.temp_ch))
                if curr_res in attn_resolutions: # 当分辨率在指定分辨率列表时，添加注意力层
                    self.downsamples.add_module('attn_block_l{}b{}'.format(i,j), AttnBlock(in_channels=block_out))
                block_in = block_out # 确保后面的残差块输入深度和输出深度相等
            # 下采样块
            if i != self.m-1:
                '''此处不是很理解为什么最后一层不加入下采样块'''
                self.downsamples.add_module('down_block_l{}'.format(i), DownsampleBlock(in_channels=block_out, with_conv=True))
                curr_res = curr_res//2
        self.C2 = block_out

        # 后处理
        self.mid = nn.Sequential(
            ResidualBlock(in_channels=self.C2, out_channels=self.C2, dropout=dropout, temb_channels=self.temp_ch),
            AttnBlock(self.C2),
            ResidualBlock(in_channels=self.C2, out_channels=self.C2, dropout=dropout, temb_channels=self.temp_ch)
        )
        self.norm_out = nn.Sequential(
            Normalize(self.C2),
            nonlinearity(),
            nn.Conv2d(self.C2, self.nz, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # H*W*C --> Conv2D --> H*W*C'
        out = self.conv_in(x)
        # m*{Residual Block, Downsample Block} --> h*w*C''
        out = self.downsamples(out)
        # Residual Block, Non-Local Block, Residual Block --> h*w*C''
        out = self.mid(out)
        # GroupNorm, Swish, Conv2D --> h*w*nz
        out = self.norm_out(out)
        return out

# 解码器
class Decoder(nn.Module):
    def __init__(self, z_channels, C1_channels, out_channels, resolution, layer_res_blocks_num, attn_resolutions, ch_mult=(1,2,4,8), dropout=False, **ignore_kwargs):
        '''resolution：生成图像分辨率'''
        super().__init__()
        self.nz = z_channels
        self.C1 = C1_channels
        self.C = out_channels
        self.m = len(ch_mult) # 上采样层数
        self.layer_res_blocks_num = layer_res_blocks_num # 每层上采样模块中含的串行残差块数量
        self.temp_ch = 0 # 残差块额外信息维度

        # 输入卷积层
        self.C2 = C1_channels*ch_mult[self.m-1] # 第一层残差块的输入深度
        self.conv_in = nn.Conv2d(z_channels, self.C2, kernel_size=3, stride=1, padding=1)

        # 前处理
        self.mid = nn.Sequential(
            ResidualBlock(in_channels=self.C2, out_channels=self.C2, dropout=dropout, temb_channels=self.temp_ch),
            AttnBlock(self.C2),
            ResidualBlock(in_channels=self.C2, out_channels=self.C2, dropout=dropout, temb_channels=self.temp_ch)
        )

        # 上采样模块（m-1倍）
        curr_res = resolution // 2**(self.m-1) # 记录实时的图像分辨率
        block_in = self.C2 # 初始输入深度

        self.upsamples = nn.Sequential()
        for i in reversed(range(self.m)): # 逐层添加残差块
            block_out = C1_channels*ch_mult[i] # 每层的输出深度
            for j in range(self.layer_res_blocks_num+1):
                self.upsamples.add_module('res_block_l{}b{}'.format(i,j), ResidualBlock(in_channels=block_in, out_channels=block_out, dropout=dropout, temb_channels=self.temp_ch))
                if curr_res in attn_resolutions: # 当分辨率在指定分辨率列表时，添加注意力层
                    self.upsamples.add_module('attn_block_l{}b{}'.format(i,j), AttnBlock(in_channels=block_out))
                block_in = block_out # 确保后面的残差块输入深度和输出深度相等
            # 下采样块
            if i != 0:
                self.upsamples.add_module('up_block_l{}'.format(i), UpsampleBlock(in_channels=block_out, with_conv=True))
                curr_res = curr_res * 2
        
        # 后处理
        self.norm_out = nn.Sequential(
            Normalize(self.C1),
            nonlinearity(),
            nn.Conv2d(self.C1, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, give_pre_end=False):
        # h*w*nz --> Conv2D --> h*w*C''
        out = self.conv_in(x)
        # Residual Block, Non-Local Block, Residual Block --> h*w*C''
        out = self.mid(out)
        # m*{Residual Block, Upsample Block} --> H*W*C'
        out = self.upsamples(out)
        if not give_pre_end:
            # GroupNorm, Swish, Conv2D --> H*W*C
            out = self.norm_out(out)
        return out





