import torch
import torch.nn as nn
from patch_embedding import PatchEmbedding
from encoder_layer import EncoderLayer
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    """
    def __init__(self, img_size=32, patch_size=4, num_classes=100,
                 d_model=128, num_heads=8, num_layers=6, d_ff=512,
                 dropout=0.1, in_channels=3):
        super(VisionTransformer, self).__init__()

        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(img_size, patch_size, d_model, in_channels)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding
        num_patches = self.patch_embedding.num_patches
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize the parameters
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: Input image tensor of shape [batch_size, in_channels, img_size, img_size]
        """
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embedding(x)  # Shape: [batch_size, num_patches, d_model]

        # Expand the class token to the batch size and concatenate
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [batch_size, num_patches + 1, d_model]

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Apply dropout
        x = self.dropout(x)

        # Pass through Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Layer normalization
        x = self.norm(x)

        # Classification head (take the representation of the [CLS] token)
        cls_output = x[:, 0]  # Shape: [batch_size, d_model]
        logits = self.fc(cls_output)  # Shape: [batch_size, num_classes]
        return logits
