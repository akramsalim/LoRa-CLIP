# clip/projection_head.py

import torch.nn as nn
from .config import Config

class ProjectionHead(nn.Module):
    """
    Maps image and text embeddings into a shared space using a linear layer.
    """
    def __init__(self, embedding_dim, projection_dim=Config().projection_size, dropout=Config().dropout):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        x = self.projection(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.layer_norm(x)
