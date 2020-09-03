import torch


def init_mask(x):
    return torch.ones(x.size()[:-1], dtype=torch.uint8, device=x.device)
