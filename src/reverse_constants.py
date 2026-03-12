import torch

def generate_reverse_sampling_constants(betas, alpha_bar):
    alpha_bar_prev = torch.cat([torch.ones(1, device=alpha_bar.device), alpha_bar[:-1]], dim=0)

    posterior_var = betas * (1.0 - alpha_bar_prev)/(1.0 - alpha_bar)
    posterior_var = torch.clamp(posterior_var, min=1e-20)

    reverse_constants = {
        "alpha_bar": alpha_bar,
        "posterior_var": posterior_var
    }

    return reverse_constants