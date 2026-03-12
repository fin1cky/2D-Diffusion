import torch

def generate_q_sample(x0, t, eps, alpha_bar):
    alpha_bar_t = alpha_bar[t].unsqueeze(1)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps

    return xt

@torch.no_grad()
def per_step_sample_ddpm(model, xt, t, betas, alphas, alpha_bar, posterior_var):
    B = xt.shape[0]
    t_int = t
    t = torch.full((B,), t, device=xt.device, dtype=torch.long)
    eps_hat = model(xt, t)

    beta_t = betas[t].unsqueeze(1)
    alpha_t = alphas[t].unsqueeze(1)
    alpha_bar_t = alpha_bar[t].unsqueeze(1)

    mu = (xt - (beta_t /torch.sqrt(1.0 - alpha_bar_t))* eps_hat)/(torch.sqrt(alpha_t))

    if t_int==0:
        return mu

    z = torch.randn_like(xt)
    sigma = torch.sqrt(posterior_var[t].unsqueeze(1))

    return mu + sigma * z

@torch.no_grad()
def sample_ddpm(model, xt, t, betas, alphas, alpha_bar, posterior_var, T=1000, n=5000, device="cpu"):
    model.eval()
    x = torch.randn(n, 2, device=device)
    for t in reversed(range(T)):
        x = per_step_sample_ddpm(model, xt, t, betas, alphas, alpha_bar, posterior_var, x, t)

    return x