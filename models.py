import torch
import torch.nn as nn
import torch.nn.functional as F

class BinActiveSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(input >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1.0] = 0
        return grad_input

def binary_activation(input):
    return BinActiveSTE.apply(input)

def binarize_weight(weight):
    alpha = weight.abs().mean(dim=list(range(1, weight.dim())), keepdim=True)
    binary_weight = torch.where(weight >= 0, 1.0, -1.0) * alpha
    return (binary_weight - weight).detach() + weight

class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(BinaryConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        bw = binarize_weight(self.weight)
        return F.conv2d(input, bw, self.bias, self.stride, self.padding)

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        bw = binarize_weight(self.weight)
        return F.linear(input, bw, self.bias)

class BinaryLayerWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_pool=False):
        super(BinaryLayerWrapper, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = BinaryConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.use_pool = use_pool
        if self.use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = binary_activation(x)
        x = self.conv(x)
        if self.use_pool:
            x = self.pool(x)
        return x

class XNORNet(nn.Module):
    def __init__(self, num_classes=10):
        super(XNORNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.features = nn.Sequential(
            BinaryLayerWrapper(128, 128, use_pool=False),
            BinaryLayerWrapper(128, 128, use_pool=True),
            BinaryLayerWrapper(128, 256, use_pool=False),
            BinaryLayerWrapper(256, 256, use_pool=True),
            BinaryLayerWrapper(256, 512, use_pool=False),
            BinaryLayerWrapper(512, 512, use_pool=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(model_type='xnor', num_classes=10):
    if model_type == 'xnor':
        return XNORNet(num_classes)
    else:
        return SimpleNet(num_classes)

class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
