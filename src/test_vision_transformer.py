import torch
from vision_transformer import VisionTransformer

# Test parameters
img_size = 32  # Size of the input image (img_size x img_size)
patch_size = 4  # Size of each patch
num_classes = 100  # Number of classes for classification
d_model = 128  # Dimensionality of the model
num_heads = 8  # Number of attention heads
num_layers = 6  # Number of encoder layers
d_ff = 512  # Dimensionality of the feed-forward network
batch_size = 4  # Number of samples in a batch
in_channels = 3  # Number of input channels (e.g., 3 for RGB images)

# Create a VisionTransformer instance
vit = VisionTransformer(
    img_size=img_size, 
    patch_size=patch_size, 
    num_classes=num_classes, 
    d_model=d_model, 
    num_heads=num_heads, 
    num_layers=num_layers, 
    d_ff=d_ff, 
    dropout=0.1, 
    in_channels=in_channels
)

# Create a dummy input tensor with shape [batch_size, in_channels, img_size, img_size]
dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)

# Compute the output using the Vision Transformer
output = vit(dummy_input)

# Print the input and output shapes
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")  # Expected: [batch_size, num_classes]

# Check that the output shape matches the expected shape
assert output.shape == (batch_size, num_classes), "Output shape does not match expected shape!"
print("Test passed: Output shape matches expected shape.")
