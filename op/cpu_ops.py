import torch
import torch.nn as nn
import torch.nn.functional as F

class CPULeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        out = F.leaky_relu(input + self.bias, negative_slope=self.negative_slope)
        return out * self.scale

def cpu_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        out = F.leaky_relu(input + bias, negative_slope=negative_slope)
    else:
        out = F.leaky_relu(input, negative_slope=negative_slope)
    return out * scale if scale != 1 else out

def cpu_upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    # Simplified upfirdn2d for CPU
    # This is a basic implementation that works for common cases
    out = input
    
    if up > 1:
        out = F.interpolate(out, scale_factor=up, mode='nearest')
    
    if len(pad) > 0:
        pad_sizes = [pad[0], pad[1], pad[0], pad[1]]
        out = F.pad(out, pad_sizes, mode='reflect')
    
    if down > 1:
        out = F.avg_pool2d(out, down, stride=down)
    
    return out
