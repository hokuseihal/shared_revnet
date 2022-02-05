import torch
import torch.nn as nn
import torch.nn.functional as F

from model import utils as MU
from model.reversible import _Reversible


class ResBlock(nn.Module):
    def __init__(self, in_ch, embch, dropout, outch, group, isclsemb, activate):
        super(ResBlock, self).__init__()
        outch = in_ch if outch is None else outch
        # self.isclsemb = isclsemb
        self.norm1 = nn.GroupNorm(group, in_ch)
        self.conv1 = nn.Conv2d(in_ch, outch, 3, 1, 1)
        # self.emb_proj = nn.Linear(embch, outch)
        self.norm2 = nn.GroupNorm(group, outch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(outch, outch, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_ch, outch, 1, 1, 0)
        self.activate = activate

    def forward(self, x, emb):
        _x = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        # emb = self.emb_proj(self.activate(emb))[:, :, None, None]
        # if self.isclsemb:
        #     temb, cemb = emb.chunk(2)
        #     x = (1 + temb) * x + cemb
        # else:
        #     x = x + emb
        x = self.norm2(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + self.shortcut(_x)


class AttnBlock(nn.Module):
    def __init__(self, feature, nhead=4):
        super(AttnBlock, self).__init__()
        self.feature = torch.tensor(feature)
        self.v = nn.Conv2d(feature, feature, 1)
        self.q = nn.Conv2d(feature, feature, 1)
        self.k = nn.Conv2d(feature, feature, 1)
        self.out = nn.Conv2d(feature, feature, 1)
        # self.attn = nn.TransformerEncoderLayer(feature, nhead, activation='gelu', dim_feedforward=dimff)
        self.norm = nn.GroupNorm(32, feature)

    def forward(self, x, emb=None):
        _x = x
        x = self.norm(x)
        return _x + self.out(torch.einsum('bhwnm,bcnm->bchw', F.softmax(
            torch.einsum('bchw,bcij->bhwij', self.q(x), self.k(x)) / self.feature.float().sqrt()), self.v(x)))
class PixelShuffle(nn.)

class ReversibleUNet(nn.Module):
    def __init__(self,
                 in_ch, feature, embch, size, bottle_attn, activate, attn_res=(), chs=(1, 2, 4),
                 num_res_block=1,
                 dropout=0, group=32,
                 isclsemb=False, out_ch=3
                 ):
        super(ReversibleUNet, self).__init__()
        if activate is None:
            activate = nn.Hardswish()
        elif activate == 'silu':
            activate = nn.SiLU()
        # self.emb = nn.Sequential(
        #     nn.Conv1d(embch, embch, 1, groups=2 if isclsemb else 1),
        #     activate,
        #     nn.Conv1d(embch, embch, 1, groups=2 if isclsemb else 1),
        # )
        _res_ch = lambda ch, outch=None: ResBlock(in_ch=ch, outch=outch, embch=embch, activate=activate, group=group,
                                                  isclsemb=isclsemb, dropout=dropout)
        self.convin = nn.Conv2d(in_ch, feature, 3, 1, 1)
        self.bottle = MU.deeplist2module([
            [_res_ch(feature * chs[-1]), _res_ch(feature * chs[-1])],
            [_res_ch(feature * chs[-1]), _res_ch(feature * chs[-1])]
        ])
        if bottle_attn:
            self.bottle.insert(1, [AttnBlock(feature * chs[-1]), AttnBlock(feature * chs[-1])])
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        prech = 1
        res = size
        for ch in chs:
            _down = []
            _up = []
            if res in attn_res: _up.append([AttnBlock(feature * prech), AttnBlock(feature * prech)])
            for idx in range(num_res_block)[::-1]:
                _down.append([_res_ch(feature * prech, feature * ch), _res_ch(feature * prech, feature * ch)])
                _up.insert(0, [_res_ch(feature * ch * (2 if idx == 0 else 1), feature * prech),
                               _res_ch(feature * ch * (2 if idx == 0 else 1), feature * prech)])
                prech = ch
            if res in attn_res: _down.append(AttnBlock(feature * ch))
            self.down.append(MU.deeplist2module(_down))
            self.up.insert(0, MU.deeplist2module(_up))
            res //= 2
        self.out = nn.Sequential(
            nn.GroupNorm(group, feature * chs[0]),
            activate,
            nn.Conv2d(feature * chs[0], out_ch, 3, 1, 1)
        )
        self.innerlayer = [AttnBlock, ResBlock]

    def forward(self, x, emb):
        emb = self.emb(emb[:, :, None])[:, :, 0]
        x0, x1 = torch.chunk(x, chunks=2, dim=1)
        output = []
        for block in self.down:
            for layer0, layer1 in block:
                assert type(layer0) == type(layer1)
                if type(layer0) in self.innerlayer:
                    x0, x1 = _Reversible._forward(x0, x1, layer0, layer1)
            output.append(x1)
            x0, x1 = pixelshuffle(x0)

        for layer0, layer1 in self.bottle:
            x0, x1 = _Reversible._forward(x0, x1, layer0, layer1)
        for block in self.up:
            x0 = torch.cat([x0, x1], dim=1)
            x1 = output.pop(-1)
            for layer0, layer1 in block:
                assert type(layer0) == type(layer1)
                x0, x1 = _Reversible._forward(x0, x1, layer0, layer1)
        assert len(output) == 0

        return torch.cat([x0, x1], dim=1)
