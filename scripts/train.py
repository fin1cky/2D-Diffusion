import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

from src.schedule import generate_schedule
from src.model import EpsMLP
from src.datasets_2d import get_sampler
from src.ddpm import generate_q_sample
from src.viz import plot_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 2D DDPM model")
    parser.add_argument("--dataset", type=str, default="circle",
                        choices=["circle", "moons", "spiral"])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--save-weights", action=argparse.BooleanOptionalAction, default=True,
                        help="Save trained model weights to model_{dataset}.pt")
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

    T = 1000
    schedule = generate_schedule(T=T)
    alpha_bar = schedule["alpha_bar"].to(device)

    sampler = get_sampler(args.dataset)
    noise = 0.05
    if args.dataset == "spiral":
        noise = 0.02

    model = EpsMLP(time_dim=128, hidden=256, depth=4).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    model.train()
    losses = []
    for step in range(1, args.steps + 1):
        x0 = sampler(args.batch_size, noise=noise, device=device)
        t = torch.randint(0, T, (args.batch_size,), device=device, dtype=torch.long)
        eps = torch.randn_like(x0)

        xt = generate_q_sample(x0, t, eps, alpha_bar)
        eps_hat = model(xt, t)
        loss = mse(eps_hat, eps)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step % args.print_every == 0:
            avg = sum(losses[-args.print_every:]) / args.print_every
            print(f"step {step:6d} | avg loss {avg:.4f}")

    if args.save_weights:
        weights_path = f"model_{args.dataset}.pt"
        torch.save(model.state_dict(), weights_path)
        print(f"model saved to {weights_path}")

    if args.save_plots:
        plot_loss(losses, args.dataset)


if __name__ == "__main__":
    main()
