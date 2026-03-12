import math
import torch

def sample_circle(B, noise=0.05, device="cpu"):
    theta = 2*math.pi*torch.rand(B, device=device)
    r = 1.0 + noise*torch.randn(B, device=device)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.stack([x, y], dim=1)

def sample_moons(B, noise=0.05, device="cpu"):
    B1 = B//2
    B2 = B - B1

    t1 = math.pi * torch.rand(B1, device=device)
    x1 = torch.cos(t1)
    y1 = torch.sin(t1)

    t2 = math.pi * torch.rand(B2, device=device)
    x2 = 1.0 - torch.cos(t2)
    y2 = - torch.sin(t2) - 0.5

    x = torch.cat([x1, x2], dim=0)
    y = torch.cat([y1, y2], dim=0)

    pts = torch.stack([x, y], dim=1)
    pts = pts + noise * torch.randn_like(pts)

    return pts

def sample_spiral(B, noise=0.02, device="cpu", t_max=8*math.pi):
    u = torch.rand(B, device=device)
    t = t_max * torch.sqrt(u)
    r = t / t_max
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    pts = torch.stack([x, y], dim=1)
    pts = pts + noise * torch.randn_like(pts)

    return pts

SAMPLERS = {
    "circle": sample_circle,
    "moons": sample_moons,
    "spiral": sample_spiral
}

def get_sampler(name: str):
    key = name.lower().strip()
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler: {name}. Choose from {list(SAMPLERS.keys())}")
    return SAMPLERS[key]
