import os
import torch
import matplotlib.pyplot as plt

from src.ddpm import generate_q_sample


def plot_schedule(alpha_bar, save_path="plots/schedule.png"):
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(alpha_bar.detach().cpu().numpy())
    ax.set_title("Cosine schedule: alpha_bar")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("alpha_bar")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_datasets(samplers, B=4096, noise=0.05, device="cpu", save_path="plots/datasets.png"):
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, len(samplers), figsize=(12, 4))
    for ax, (name, fn) in zip(axes, samplers.items()):
        x0 = fn(B, noise=noise, device=device).detach().cpu()
        ax.scatter(x0[:, 0], x0[:, 1], s=2)
        ax.set_title(name)
        ax.axis("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_forward_diffusion(samplers, alpha_bar, B=4096, timesteps=None, noise=0.05,
                           device="cpu", save_path="plots/forward_diffusion.png"):
    if timesteps is None:
        timesteps = [0, 99, 499, 999]
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(len(samplers), len(timesteps), figsize=(12, 10))
    for row, (name, fn) in enumerate(samplers.items()):
        x0 = fn(B, noise=noise, device=device)
        for col, tv in enumerate(timesteps):
            t = torch.full((B,), tv, device=device, dtype=torch.long)
            eps = torch.randn_like(x0)
            xt = generate_q_sample(x0, t, eps, alpha_bar).detach().cpu()
            ax = axes[row, col]
            ax.scatter(xt[:, 0], xt[:, 1], s=2)
            ax.set_title(f"{name}, t={tv}")
            ax.axis("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_loss(losses, dataset_name, save_path=None):
    if save_path is None:
        save_path = f"plots/loss_{dataset_name}.png"
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_title(f"Training loss ({dataset_name})")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_samples(x, title="samples", save_path=None):
    if save_path is None:
        slug = title.lower().replace(" ", "_").replace(":", "")
        save_path = f"plots/{slug}.png"
    os.makedirs("plots", exist_ok=True)
    x = x.detach().cpu()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x[:, 0], x[:, 1], s=2)
    ax.axis("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_gamma_sweep(model, betas, alphas, alpha_bar, posterior_var, dataset_name,
                     gammas=None, n=5000, T=1000, device="cpu", save_path=None):
    if gammas is None:
        gammas = [1.0, 0.5, 0.2, 0.0]
    if save_path is None:
        save_path = f"plots/gamma_sweep_{dataset_name}.png"
    os.makedirs("plots", exist_ok=True)

    @torch.no_grad()
    def _sample_noise_scaled(gamma):
        model.eval()
        x = torch.randn(n, 2, device=device)
        for t_int in reversed(range(T)):
            B = x.shape[0]
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)
            eps_hat = model(x, t_batch)

            beta_t = betas[t_batch].unsqueeze(1)
            alpha_t = alphas[t_batch].unsqueeze(1)
            alpha_bar_t = alpha_bar[t_batch].unsqueeze(1)

            mu = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat) / torch.sqrt(alpha_t)

            if t_int == 0:
                x = mu
            else:
                z = torch.randn_like(x)
                sigma = gamma * torch.sqrt(posterior_var[t_batch].unsqueeze(1))
                x = mu + sigma * z
        return x

    fig, axes = plt.subplots(1, len(gammas), figsize=(16, 4))
    for ax, g in zip(axes, gammas):
        xg = _sample_noise_scaled(g).detach().cpu()
        ax.scatter(xg[:, 0], xg[:, 1], s=2)
        ax.set_title(f"gamma={g}")
        ax.axis("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(f"DDPM gamma sweep: {dataset_name}")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
