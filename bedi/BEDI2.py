import torch
import torch.nn as nn
from sympy.strategies.branch import condition
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import math


class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, beta=1):
        super(ECAModule, self).__init__()

        # 计算自适应卷积核大小
        kernel_size = int(abs(math.log(channels, 2) + beta) / gamma)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LayerNorm(nn.Module):
    """
    channels_last corresponds to inputs with shape (batch_size, height, width, channels)
    channels_first corresponds to inputs with shape (batch_size, channels, height, width)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, groups=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim

        # 使用单个卷积层，支持分组卷积
        self.conv = nn.Conv2d(
            inp_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=bias
        )

        # 批归一化层
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

        # 激活函数
        self.relu = None
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


class IRMLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=32, expansion=4):
        super().__init__()
        hidden = in_dim * expansion     # 仍按输入宽度扩张

        # 主分支 -------------------------------------------------------------
        self.expand  = Conv(in_dim, hidden, 1, bias=False)
        self.act1    = nn.GELU()
        self.dwconv  = Conv(hidden, hidden, 3, groups=hidden, bias=False)
        self.act2    = nn.GELU()
        self.project = Conv(hidden, out_dim, 1, bias=False)
        self.bn      = nn.BatchNorm2d(out_dim)

        # skip 分支：只有当 in_dim ≠ out_dim 时才需要投影
        self.shortcut = (
            nn.Identity()
            if in_dim == out_dim
            else Conv(in_dim, out_dim, 1, bias=False)
        )

    def forward(self, x):
        id = self.shortcut(x)           # (B, out_dim, H, W)

        x = self.expand(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.project(x)
        x = self.bn(x)
        return x + id                   # 输出 (B, out_dim, H, W)




class DynamicConv2d(nn.Module):
    """
    CondConv‑style dynamic convolution layer.

    Each sample in the batch receives its own kernel,
    computed as a weighted sum of `num_experts` static expert kernels.
    The routing (gating) weights are produced from a global‑average‑pooled
    summary of the same input.

    Args
    ----
    in_channels  : int
    out_channels : int
    kernel_size  : int or tuple
    stride       : int or tuple, default 1
    padding      : int or tuple, default 0
    dilation     : int or tuple, default 1
    groups       : int, default 1
    bias         : bool, default True
    num_experts  : int, number of expert kernels (M in the paper)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        num_experts=4,
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts

        # M expert kernels and (optionally) biases
        weight_shape = (
            num_experts,
            out_channels,
            in_channels // groups,
            *self.kernel_size,
        )
        self.weight = nn.Parameter(torch.randn(*weight_shape) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, out_channels))
        else:
            self.bias = None

        # Routing network: GAP ➜ linear ➜ softmax  (produces M scalar weights / sample)
        self.gate = nn.Linear(in_channels, num_experts)

    def _combine_kernels(self, alpha):
        """
        alpha: (B, M) mixing coefficients after softmax.
        Return: list of B kernels, each (out_C, in_C/groups, k, k)
        """
        B = alpha.size(0)
        # alpha.unsqueeze(2..) broadcasts over kernel dims
        # (B, M, 1, 1, 1, 1) * (M, outC, inC/groups, k, k)
        combined = torch.sum(
            alpha[:, :, None, None, None, None] * self.weight, dim=1
        )
        return combined

    def forward(self, x):
        B, C, H, W = x.shape

        # ---- 1. Routing / gating weights -----------------------------------
        # Global average pooling (B, C, 1, 1) -> (B, C)
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)
        alpha = F.softmax(self.gate(gap), dim=1)  # (B, M)

        # ---- 2. Combine expert kernels for each sample ----------------------
        kernels = self._combine_kernels(alpha)  # (B, outC, inC/groups, k, k)
        if self.bias is not None:
            biases = torch.sum(alpha[:, :, None] * self.bias, dim=1)  # (B, outC)
        else:
            biases = [None] * B  # placeholder iterable

        # ---- 3. Convolution (sample‑wise loop for clarity) ------------------
        outputs = []
        for i in range(B):
            out_i = F.conv2d(
                x[i : i + 1],
                kernels[i],
                bias=None if self.bias is None else biases[i],
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            outputs.append(out_i)
        return torch.cat(outputs, dim=0)






# Hierachical Feature Fusion Block
class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2,ch_int0,ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim0= Conv(ch_int0//2, ch_int0, 1, bn=True, relu=True)
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.dynamic_conv = DynamicConv2d(ch_1, ch_int,kernel_size=3, padding=1, num_experts=1)

        self.ECA = ECAModule(channels=ch_int)

        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int,ch_int,expansion=1)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):
        g = self.Updim(g)
        g = self.Avg(g)
        W_local = self.W_l(l)#(1,192,28,28) # local feature from Local Feature Block
        W_global = self.W_g(g)#(1,192,28,28) # global feature from Global Feature Block
        if f is not None:
            f = self.Updim0(f)
            W_f = self.Updim(f)#(1,192,56,56)
            W_f = self.Avg(W_f)#(1,192,28,28)
            W_f = self.Avg(W_f)
            shortcut = W_f#(1,192,28,28)
            X_f = torch.cat([W_f, W_local, W_global], 1)#(1,576,28,28)
            X_f = self.norm1(X_f)#(1,576,28,28)
            X_f = self.W3(X_f)#(1,192,28,28)
            X_f = self.gelu(X_f)#(1,192,28,28)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)


        l = self.dynamic_conv(l)

        g = self.ECA(g)

        fuse = torch.cat([g, l, X_f], 1)#(1,576.28,28)
        fuse = self.norm3(fuse)#(1,576,28,28)
        fuse = self.residual(fuse)#(1,192,28,28)
        # fuse = shortcut + self.drop_path(fuse)#(1,192,28,28)
        return fuse




# 定义 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # 定义两个 3x3 卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# 定义 ResNet 主架构
class BEDI(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(BEDI, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 主干的 4 个阶段
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])

        # 上采样模块，将 layer3 的输出通道数调整为 512
        self.upsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),  # 调整通道数到 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fuse = HFF_block(ch_1=256,ch_2=256,r_2=16,ch_int0=128,ch_int=256,ch_out=256,drop_rate=0)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 初始化残差分支的最后一个 BatchNorm 层
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#(1,64,56,56)

        x1 = self.layer1(x)#(1,64,56,56)f
        x2 = self.layer2(x1)#(1,128,28,28)g
        x3 = self.layer3(x2)#(1,256,14,14)l
        f = self.fuse(x3, x2, x1)
        x = torch.torch.cat([x3, f], 1)#(1,512,14,14)
        # x = self.layer4(x)
        # f = self.fuse(x3,x2,x1)


        x = self.avgpool(x)#(1,512,1,1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ResNet 官方预训练模型权重 URL
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
}



# 构造 ResNet-18 的辅助函数
def _bedi(arch, block, layers, pretrained, progress, **kwargs):
    model = BEDI(block, layers, **kwargs)
    if pretrained:
        param_state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in param_state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    return model

def bedi(pretrained=False, progress=True, **kwargs):


    return _bedi('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

# 测试模型
if __name__ == '__main__':
    model = bedi(pretrained=True)  # 加载带有预训练权重的 ResNet-50
    print(model)

    # 测试模型的前向传播
    x = torch.randn(1, 3, 224, 224)  # 输入一个 224x224 的随机图像
    output = model(x)
    print(output.shape)
