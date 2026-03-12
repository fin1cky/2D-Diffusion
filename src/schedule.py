import math
import torch

def generate_alpha_bar_t(T=1000, s=0.008):
   t = torch.arange(T+1).float()
   x = (t/T + s)/(1 + s)
   angles = x * ((math.pi)/2)
   f = torch.cos(angles) ** 2
   alpha_bar_t = f/f[0]
   
   return alpha_bar_t

def generate_schedule(T=1000, s=0.008):
    alpha_bar_t = generate_alpha_bar_t(T, s)
    betas = 1.0 - (alpha_bar_t[1:])/(alpha_bar_t[:-1])
    betas = torch.clamp(betas, 1e-5, 0.999)

    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    schedule = {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_t": alpha_bar_t
    }

    return schedule