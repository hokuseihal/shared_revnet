import torch


class _Reversible:
    @staticmethod
    @torch.no_grad()
    def _forward(x0, x1, layer0, layer1):
        z1 = layer0(x0) + x1
        return x0 + layer1(z1), z1

    @staticmethod
    def _backward(y0, y1, dy0, dy1, layer0, layer1):
        y1.requires_grad = True
        with torch.enable_grad():
            gy1 = layer1(y1)
            gy1.backward(dy0)
        with torch.no_grad():
            x0 = y0 - gy1
        del gy1
        del y0
        x0.requires_grad = True
        with torch.enable_grad():
            fx0 = layer0(x0)
            fx0.backward(dy1 + y1.grad)
        with torch.no_grad():
            x1 = y1 - fx0
        del fx0

        return x0, x1, x0.grad + dy0, y1.grad + dy1


class _ReversivleLayers(torch.autograd.Function):

    @staticmethod
    def forward(ctx, layers, x, innerlayers):
        ctx.layers = layers
        ctx.innerlayers = innerlayers
        x0, x1 = torch.chunk(x, chunks=2, dim=1)
        del x
        for layer0, layer1 in layers:
            assert type(layer0) == type(layer1)
            if (type(layer0) in ctx.innerlayers):
                x0, x1 = _Reversible._forward(x0, x1, layer0, layer1)
            else:
                x0 = layer0._forward(x0)
                x1 = layer1._forward(x1)
        ctx.save_for_backward(torch.cat([x0, x1], dim=1))
        return torch.cat([x0, x1], dim=1)

    @staticmethod
    def backward(ctx, grads):
        y0, y1 = ctx.saved_tensors[0].chunk(chunks=2, dim=1)
        dy0, dy1 = grads.chunk(chunks=2, dim=1)
        del grads
        for layer0, layer1 in ctx.layers[::-1]:
            assert type(layer0) == type(layer1)
            if (type(layer0) in ctx.innerlayers):
                y0, y1, dy0, dy1 = _Reversible._backward(y0, y1, dy0, dy1, layer0, layer1)
            else:
                y0, dy0 = layer0._backward(dy0, y=y0)
                y1, dy1 = layer1._backward(dy1, y=y1)
            # print(f"{y0=},{y1=},{dy0=},{dy1=}")
        return None, torch.cat([dy0, dy1], dim=1), None


class ReversibleLayers(torch.nn.Module):
    def __init__(self, layers, innerlayers):
        super(ReversibleLayers, self).__init__()
        self.rl = _ReversivleLayers.apply
        self.layers = layers
        self.innerlayers = innerlayers

    def forward(self, x):
        return self.rl(self.layers, x, self.innerlayers)
