import torch
from position_wise_feed_forward import PositionWiseFeedForward

# Test parameters
d_model = 128  # Dimensionality of the input/output
d_ff = 512  # Dimensionality of the hidden layer
seq_length = 10  # Length of the input sequence
batch_size = 4  # Number of samples in a batch

# Create a PositionWiseFeedForward instance
ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)

# Create a dummy input tensor with shape [batch_size, seq_length, d_model]
dummy_input = torch.randn(batch_size, seq_length, d_model)

# Compute the output using the forward method
output = ffn(dummy_input)

# Print the input and output shapes
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")  # Expected: [batch_size, seq_length, d_model]

# Check that the output shape matches the input shape
assert output.shape == dummy_input.shape, "Output shape does not match input shape!"
print("Test passed: Output shape matches input shape.")