import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Feed-forward network applied at each position separately and identically.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, seq_length, d_model]
        """
        x = self.fc1(x)  # Shape: [batch_size, seq_length, d_ff]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Shape: [batch_size, seq_length, d_model]
        return x
