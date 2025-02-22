import torch
from torch import nn
import torch.nn.functional as F

# Dual-FFN 双分支卷积前馈网络
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ffn_branch1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=3, stride=1)
        )
        self.ffn_branch2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv1 = nn.Conv2d(2 * d_model, d_model, 3, padding=1)

    def forward(self, src, spatial_shapes, *args):
        split_list = [(w * h) for (w, h) in spatial_shapes]
        feat_levels = []
        for memory, (w, h) in zip(src.split(split_list, 1), spatial_shapes):
            memory = memory.view(-1, w, h, self.d_model).permute(0, 3, 1, 2)
            memory1 = self.ffn_branch1(memory)
            memory2 = self.ffn_branch2(memory)
            memory_mid = torch.cat((memory1, memory2), dim=1)
            memory = F.upsample(memory_mid, size=memory.size()[2:], mode='bilinear')
            memory = self.conv1(memory)
            feat_levels.append(memory.flatten(2).transpose(1, 2))
        return torch.cat(feat_levels, 1)


class VanillaFeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1, bias=False)
        )

    def forward(self, src, *args):
        return self.ffn(src.permute(0, 2, 1)).permute(0, 2, 1)


class StdFeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, *args):
        return self.norm(src + self.ffn(src))


def get_ffn(d_model, ffn_type):
    if ffn_type == 'std':
        return StdFeedForwardNetwork(d_model)
    if ffn_type == 'vanilla':
        return VanillaFeedForwardNetwork(d_model)
    return FeedForwardNetwork(d_model)
