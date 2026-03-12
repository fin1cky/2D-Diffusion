import argparse
import torch

from src.schedule import generate_schedule
from src.model import EpsMLP
from src.ddpm import per_step_sample_ddpm
from src.ddim import sample_ddim_deterministic
from src.reverse_constants import generate_reverse_sampling_constants
from src.viz import plot_samples, plot_gamma_sweep


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained 2D DDPM model")
    parser.add_argument("--dataset", type=str, default="circle",
                        choices=["circle", "moons", "spiral"])
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (.pt). Defaults to model_{dataset}.pt")
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of samples to generate")
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True,
                        help="Save plots to plots/ directory")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"device: {device}")

    weights_path = args.weights or f"model_{args.dataset}.pt"

    T = 1000
    schedule = generate_schedule(T=T)
    betas = schedule["betas"].to(device)
    alphas = schedule["alphas"].to(device)
    alpha_bar = schedule["alpha_bar"].to(device)

    reverse_constants = generate_reverse_sampling_constants(betas, alpha_bar)
    posterior_var = reverse_constants["posterior_var"]

    model = EpsMLP(time_dim=128, hidden=256, depth=4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {weights_path}")

    print("Running DDPM sampling...")
    x = torch.randn(args.n, 2, device=device)
    for t_int in reversed(range(T)):
        x = per_step_sample_ddpm(model, x, t_int, betas, alphas, alpha_bar, posterior_var)
    print(f"DDPM samples shape: {x.shape}, range: [{x.min():.2f}, {x.max():.2f}]")

    if args.save_plots:
        plot_samples(x, title=f"DDPM samples: {args.dataset}")

    print(f"Running DDIM sampling ({args.ddim_steps} steps)...")
    x_ddim = sample_ddim_deterministic(model, alpha_bar, T=T, n=args.n, steps=args.ddim_steps)
    print(f"DDIM samples shape: {x_ddim.shape}, range: [{x_ddim.min():.2f}, {x_ddim.max():.2f}]")

    if args.save_plots:
        plot_samples(x_ddim, title=f"DDIM samples: {args.dataset}")
        plot_gamma_sweep(model, betas, alphas, alpha_bar, posterior_var, args.dataset, device=device)


if __name__ == "__main__":
    main()
