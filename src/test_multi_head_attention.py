import torch
from multi_head_attention import MultiHeadAttention

# Test parameters
d_model = 128  # Dimensionality of the model
num_heads = 8  # Number of attention heads
seq_length = 10  # Length of the input sequence (number of tokens)
batch_size = 4  # Number of samples in a batch

# Create a MultiHeadAttention instance
multi_head_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Create dummy input tensors for queries, keys, and values
# Shape: [batch_size, seq_length, d_model]
Q = torch.randn(batch_size, seq_length, d_model)
K = torch.randn(batch_size, seq_length, d_model)
V = torch.randn(batch_size, seq_length, d_model)

# Compute the output using the forward method
output = multi_head_attn(Q, K, V)

# Print the input and output shapes
print(f"Input shape: Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
print(f"Output shape: {output.shape}")  # Expected shape: [batch_size, seq_length, d_model]

# Check that the output shape matches the input shape
assert output.shape == Q.shape, "Output shape does not match input shape!"
print("Test passed: Output shape matches input shape.")
