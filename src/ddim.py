import torch

@torch.no_grad()
def sample_ddim_deterministic(model, alpha_bar, T=1000, n=5000, steps=50):
    model.eval()
    device = alpha_bar.device
    x = torch.randn(n, 2, device=device)
    ts = torch.linspace(T-1, 0, steps, device=device).long()

    for i in range(len(ts) - 1):
        t = ts[i].item()
        s = ts[i+1].item()

        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        s_batch = torch.full((n,), s, device=device, dtype=torch.long)

        eps_hat = model(x, t_batch)
        alpha_bar_t = alpha_bar[t_batch].unsqueeze(1)

        x0_hat = (x - ((torch.sqrt(1.0 - alpha_bar_t))*eps_hat))/(torch.sqrt(alpha_bar_t))
        x0_hat = torch.clamp(x0_hat, -2.0, 2.0)

        alpha_bar_s = alpha_bar[s_batch].unsqueeze(1)

        x = (torch.sqrt(alpha_bar_s) * x0_hat) + (torch.sqrt(1.0 - alpha_bar_s) * eps_hat)

    return x