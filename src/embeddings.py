import math
import torch

def time_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    t = t.view(-1).float()
    half = dim // 2
    freqs = torch.exp(-math.log(10000)*(torch.arange(0, half, device=t.device).float()/max(half-1, 1)))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim%2==1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=1)

    return emb