import torch
import torch.nn as nn

class Affine(nn.Module):
    def __init__(self, channels, hidden=32, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 用一个两层 MLP 来预测 scale 和 shift
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2)  # 输出一半为 scale，一半为 shift
        )
        # 初始化：让输出接近恒等变换
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.02)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x : (B, L, C)
        # 在时间维上做平均池化，得到 (B, C)
        pooled = x.mean(dim=1)
        # 通过 MLP 预测 scale 和 shift
        params = self.net(pooled)  # (B, 2*C)
        scale, shift = params.chunk(2, dim=-1)
        # 调整维度为 (B, 1, C)，以便在时间维广播
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        # 输出仿射变换结果
        return x * (1.0 + scale) + shift

