import torch
import torch.nn as nn

from model import nn as _nn
from model.revnet import ReversibleLayers


def test_reversible_backward_square():
    for device in ['cpu','cuda']:
        class Square(nn.Module):
            def __init__(self):
                super(Square, self).__init__()

            def forward(self, x):
                return x ** 2

        layers = [[Square(), Square()],]
        rl = ReversibleLayers(layers, [Square])
        x = torch.tensor([[2., 3.]], requires_grad=True, device=device)
        y = rl(x)
        y.sum().backward()
        print(f"{y=}")
        print(f"{x.grad=}")
        gt = torch.tensor([61, 15], device=device)
        assert (x.grad - gt).abs().mean() < 1e-7


def base_reversible_backward_ident(device):
    layers = [
        [nn.Identity(), nn.Identity()],
        [nn.Identity(), nn.Identity()],
        [nn.Identity(), nn.Identity()],
        [nn.Identity(), nn.Identity()],
        [nn.Identity(), nn.Identity()],
    ]
    rl = ReversibleLayers(layers, [nn.Identity])
    x = torch.ones(1, 8, requires_grad=True, device=device)
    y = rl(x)
    y.sum().backward()
    # print(f"{y=}")
    # print(f"{x.grad=}")
    gt = torch.tensor([144, 144, 144, 144, 89, 89, 89, 89], device=device)
    assert (x.grad - gt).abs().mean() < 1e-7


def base_reversible_backward_upsample(device):
    layers = [
        [nn.Identity(), nn.Identity()],
        [_nn.Upsample(scale_factor=2), _nn.Upsample(scale_factor=2)],
        [nn.Identity(), nn.Identity()],
        [_nn.Upsample(scale_factor=2), _nn.Upsample(scale_factor=2)],
    ]
    rl = ReversibleLayers(layers, [nn.Identity])
    x = torch.ones(1, 2, 1, 1, requires_grad=True, device=device)
    y = rl(x)
    y.sum().backward()
    print(f"{y=}")
    print(f"{x.grad=}")
    gt = torch.tensor([[[[32.]], [[20.]]]], device=device)
    assert (x.grad - gt).abs().mean() < 1e-7


def test_reversible_backward_upsample():
    base_reversible_backward_upsample('cpu')


def test_reversible_backward_upsample_cuda():
    base_reversible_backward_upsample('cuda')


def base_reversible_backward_channelchanger(device):
    layers = [
        [nn.Identity(), nn.Identity()],
        [_nn.Upsample(scale_factor=2), _nn.Upsample(scale_factor=2)],
        [_nn.ChannelChanger(outch=4), _nn.ChannelChanger(outch=4)],
        [nn.Identity(), nn.Identity()],
        [_nn.Upsample(scale_factor=2), _nn.Upsample(scale_factor=2)],
        [_nn.ChannelChanger(outch=8), _nn.ChannelChanger(outch=8)]
    ]
    rl = ReversibleLayers(layers, [nn.Identity])
    x = torch.ones(1, 2, 1, 1, requires_grad=True, device=device)
    y = rl(x)
    y.sum().backward()
    print(f"{y=}")
    print(f"{x.grad=}")
    gt = torch.tensor([[[[32.]], [[20.]]]], device=device)
    assert (x.grad - gt).abs().mean() < 1e-7


def test_reversible_backward_channelchanger():
    base_reversible_backward_channelchanger('cpu')


def test_reversible_backward_channelchanger_cuda():
    base_reversible_backward_channelchanger('cuda')


def test_reversible_ident():
    base_reversible_backward_ident('cpu')


def test_reversible_ident_cuda():
    base_reversible_backward_ident('cuda')


def base_reversible_linear(device):
    layers = []
    for _ in range(1):
        l0 = nn.Linear(1, 1)
        l0.weight = nn.Parameter(torch.ones_like(l0.weight, device=device) * 2)
        l0.bias = nn.Parameter(torch.ones_like(l0.bias, device=device) * 2)
        l1 = nn.Linear(1, 1)
        l1.weight = nn.Parameter(torch.ones_like(l1.weight, device=device) * 2)
        l1.bias = nn.Parameter(torch.ones_like(l1.bias, device=device) * 2)
        layers.append([l0, l1])
    rl = ReversibleLayers(layers, [nn.Linear])
    x = torch.ones(1, 2, requires_grad=True, device=device)
    y = rl(x)
    y.sum().backward()
    print(f"{y=}")
    print(f"{x.grad=}")
    print(f'{layers[0][0].weight.grad=}')
    print(f'{layers[0][0].bias.grad=}')
    print(f'{layers[0][1].weight.grad=}')
    print(f'{layers[0][1].bias.grad=}')
    gt_x_grad = torch.tensor([7, 3], device=device)
    gt_fw_grad = torch.tensor([3], device=device)
    gt_fb_grad = torch.tensor([3], device=device)
    gt_gw_grad = torch.tensor([5], device=device)
    gt_gb_grad = torch.tensor([1], device=device)
    assert (x.grad - gt_x_grad).abs().mean() < 1e-7
    assert (layers[0][0].weight.grad - gt_fw_grad).abs().mean() < 1e-7
    assert (layers[0][0].bias.grad - gt_fb_grad).abs().mean() < 1e-7
    assert (layers[0][1].weight.grad - gt_gw_grad).abs().mean() < 1e-7
    assert (layers[0][1].bias.grad - gt_gb_grad).abs().mean() < 1e-7


def test_reversible_linear():
    base_reversible_linear('cpu')


def test_reversible_linear_cuda():
    base_reversible_linear('cuda')


if __name__ == '__main__':
    base_reversible_backward_square('cpu')
