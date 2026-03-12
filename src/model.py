import torch
import torch.nn as nn

from src.embeddings import time_embedding

class EpsMLP(nn.Module):
    def __init__(self, time_dim=128, hidden=256, depth=4):
        super().__init__()
        self.time_dim = time_dim
        self.hidden_dim = hidden
        input_dim = 2 + time_dim
        output_dim = 2
        layers = []
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.SiLU())

        for _ in range(depth-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t):
        time_embs = time_embedding(t, self.time_dim)
        h = torch.cat([x_t, time_embs], dim=1)
        return self.net(h)