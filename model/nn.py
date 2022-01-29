import torch
import torch.nn as nn
import torch.nn.functional as F

ZERO = 'zero'


class ChannelChanger(nn.Module):
    def __init__(self, outch):
        super(ChannelChanger, self).__init__()
        self.outch = outch
    @torch.no_grad()
    def _forward(self, x):
        B, C, H, W = x.shape
        self.inch = C
        assert (self.outch>=C)
        return torch.cat([x, torch.zeros(B, self.outch-C, H, W, device=x.device)], dim=1)
    @torch.no_grad()
    def _backward(self, grad, y):
        y=y[:, :self.inch]
        grad=grad[:, :self.inch]
        return y,grad


class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    @torch.no_grad()
    def _forward(self, x):
        if (self.training):
            self.input = x
        x=F.upsample_bilinear(x, scale_factor=self.scale_factor)
        return x
    @torch.no_grad()
    def _backward(self, grad, **kwargs):
        grad=F.upsample_bilinear(grad, scale_factor=1 / self.scale_factor) * self.scale_factor
        input=self.input
        self.input=None
        return input,grad
