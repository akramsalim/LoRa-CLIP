import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    """
    Single layer of the Transformer encoder.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: Input tensor of shape [batch_size, seq_length, d_model]
        mask: Optional mask tensor.
        """
        # Self-attention sublayer with residual connection and normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward sublayer with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x
