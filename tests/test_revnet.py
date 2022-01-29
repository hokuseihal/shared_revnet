import torch
import torch.nn.functional as F

from model import modeldic


def test_revnet18():
    for device in ['cuda']:  # ['cpu','cuda']:

        m = modeldic['revnet18_r2'](num_classes=10)
        optimzer = torch.optim.Adam(m.parameters())
        m.to(device=device)
        # torch.set_grad_enabled(False)
        x = torch.randn(8, 3, 512, 512, device=device)
        y = m(x)
        loss = F.cross_entropy(y, torch.randint(8, (8,), device=device))
        loss.backward()
        optimzer.step()
        # optimzer.zero_grad()
        # if (device == 'cuda'):
        #     mem_params = sum([param.nelement() * param.element_size() for param in m.parameters()])
        #     mem_bufs = sum([buf.nelement() * buf.element_size() for buf in m.buffers()])
        #     mem = mem_params + mem_bufs
        #     print(mem // (1000 ** 2))

        for n, p in m.named_parameters():
            assert (p.grad is not None and (p.grad - 0).abs().mean() > 1e-7), f"Module {n} is not trained."


if __name__ == '__main__':
    test_revnet18()
