# mo_affine.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineExpert(nn.Module):
    """
    单个仿射专家：输入 pooled (B, N)，输出对 E 维的 scale/shift：形状 (B, E)
    """
    def __init__(self, e_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(e_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, e_dim * 2)
        )
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.02)

    def forward(self, pooled):  # pooled: (B, E)
        params = self.net(pooled)      # (B, 2E)
        scale, shift = params.chunk(2, dim=-1)
        return scale, shift            # (B, E), (B, E)


class MoAffine(nn.Module):
    """
    适配 (B, N, E) 的混合仿射专家层：
    - N 维用于 pooling 和 gating。
    - E 维是仿射作用目标（输出 scale/shift 也在 E 维）。
    """
    def __init__(self, e_dim, n_experts=1, expert_hidden=64,
                 gate_hidden=64, top_k=0, temperature=1.0):

        super().__init__()
        self.e_dim = e_dim      # 原 L
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature
        self.eps = 1e-6

        # 专家：输出 (B, E) 的 scale/shift
        self.experts = nn.ModuleList([
            AffineExpert(e_dim, hidden=e_dim // 2)
            for _ in range(n_experts)
        ])

        # gating：输入 pooled(B, E)，输出 (B, n_experts)
        self.gate = nn.Sequential(
            nn.Linear(e_dim, e_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(e_dim // 2, n_experts)
        )


    def forward(self, x):
        """
        x: (B, N, E)
        返回: (B, N, E)
        """

        B, N, E = x.shape
        assert E == self.e_dim

        # 1) 先对 N 维做池化，保留时间维信息
        pooled = x.mean(dim=1)        # (B, E)

        # 2) 逐专家输出 scale/shift
        scales = []
        shifts = []
        for expert in self.experts:
            s, sh = expert(pooled)            # (B, E)
            scales.append(s.unsqueeze(0))     # (1,B,E)
            shifts.append(sh.unsqueeze(0))    # (1,B,E)

        scales = torch.cat(scales, dim=0)     # (E, B, E)
        shifts = torch.cat(shifts, dim=0)     # (E, B, E)

        # 3) gating
        logits = self.gate(pooled)            # (B, num_experts)

        if self.temperature != 1.0:
            logits = logits / self.temperature

        gates = F.softmax(logits, dim=-1)     # (B, num_experts)

        # top-k 稀疏门控
        if self.top_k > 0:
            topk_vals, topk_idx = torch.topk(gates, self.top_k, dim=-1)
            mask = torch.zeros_like(gates)
            mask.scatter_(-1, topk_idx, 1.0)
            gates = gates * mask
            gates = gates / (gates.sum(dim=-1, keepdim=True) + self.eps)

        # 4) 聚合 scale/shift
        scales_bec = scales.permute(1, 0, 2)  # (B, E, E)
        shifts_bec = shifts.permute(1, 0, 2)

        scale = (gates.unsqueeze(-1) * scales_bec).sum(dim=1)  # (B, E)
        shift = (gates.unsqueeze(-1) * shifts_bec).sum(dim=1)  # (B, E)

        scale = scale.unsqueeze(1)   # (B,1,E)
        shift = shift.unsqueeze(1)   # (B,1,E)

        # 5) 仿射变换作用在 E 维
        out = x * (1 + scale) + shift      # 广播到 (B,N,E)
        return out