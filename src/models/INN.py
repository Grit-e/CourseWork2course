import torch
import torch.nn as nn


class STNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=100, hidden_layers=4, negative_slope=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InvertibleBlock(nn.Module):
    def __init__(self, dim, hidden_dim=100, hidden_layers=4, clamp_scale=2.0):
        super().__init__()
        assert dim % 2 == 0, "dim должен быть чётным."
        self.dim = dim
        self.d1 = dim // 2
        self.d2 = dim - self.d1
        self.clamp_scale = clamp_scale

        self.s2 = STNet(self.d2, self.d1, hidden_dim, hidden_layers)
        self.t2 = STNet(self.d2, self.d1, hidden_dim, hidden_layers)

        self.s1 = STNet(self.d1, self.d2, hidden_dim, hidden_layers)
        self.t1 = STNet(self.d1, self.d2, hidden_dim, hidden_layers)

    def _scale(self, s):
        if self.clamp_scale is None:
            return s
        return self.clamp_scale * torch.tanh(s / self.clamp_scale)

    def forward(self, u):
        u1, u2 = u[:, :self.d1], u[:, self.d1:]

        log_s2 = self._scale(self.s2(u2))
        t2 = self.t2(u2)
        v1 = u1 * torch.exp(log_s2) + t2

        log_s1 = self._scale(self.s1(v1))
        t1 = self.t1(v1)
        v2 = u2 * torch.exp(log_s1) + t1

        return torch.cat([v1, v2], dim=1)

    def inverse(self, v):
        v1, v2 = v[:, :self.d1], v[:, self.d1:]

        log_s1 = self._scale(self.s1(v1))
        t1 = self.t1(v1)
        u2 = (v2 - t1) * torch.exp(-log_s1)

        log_s2 = self._scale(self.s2(u2))
        t2 = self.t2(u2)
        u1 = (v1 - t2) * torch.exp(-log_s2)

        return torch.cat([u1, u2], dim=1)


class PermutationLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        perm = torch.randperm(dim)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x):
        return x[:, self.perm]

    def inverse(self, x):
        return x[:, self.inv_perm]


class INN(nn.Module):
    def __init__(self, dim=4, num_blocks=10, hidden_dim=100, hidden_layers=4, clamp_scale=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            InvertibleBlock(
                dim=dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                clamp_scale=clamp_scale
            )
            for _ in range(num_blocks)
        ])
        self.perms = nn.ModuleList([
            PermutationLayer(dim=dim)
            for _ in range(num_blocks - 1)
        ])

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.perms):
                x = self.perms[i](x)
        return x

    def inverse(self, y):
        for i in reversed(range(len(self.blocks))):
            if i < len(self.perms):
                y = self.perms[i].inverse(y)
            y = self.blocks[i].inverse(y)
        return y