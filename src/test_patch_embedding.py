import torch
from patch_embedding import PatchEmbedding
from torchvision import transforms
from PIL import Image

# Parameters
img_size = 32
patch_size = 4
d_model = 128
batch_size = 1  # We'll use a single image for this test
in_channels = 3

# Create a PatchEmbedding instance
patch_embedding = PatchEmbedding(img_size, patch_size, d_model, in_channels)

# Load a real image using PIL and resize it to the required size (img_size x img_size)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize the image to 32x32
    transforms.ToTensor()  # Convert the image to a tensor
])

image_path = '/home/akram/Desktop/gpu.jpeg'
image = Image.open(image_path)

# Apply the transformations and add a batch dimension
image = transform(image).unsqueeze(0)  # Shape will be [1, 3, 32, 32]

# Ensure the input tensor shape is [batch_size, in_channels, img_size, img_size]
print(f"Input shape: {image.shape}")  # Expected: [1, 3, 32, 32]

# Get the output from the PatchEmbedding
output = patch_embedding(image)

# Print shapes to verify correctness
print(f"Output shape: {output.shape}")  # Expected: [1, 64, 128]