# clip/clip_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .projection_head import ProjectionHead
from .config import Config

class CLIPModel(nn.Module):
    """
    CLIP model that aligns image and text embeddings using contrastive learning.
    """
    def __init__(self):
        super(CLIPModel, self).__init__()
        config = Config()
        self.image_encoder = ImageEncoder(
            model_name=config.image_model,
            pretrained=config.use_pretrained,
            trainable=config.fine_tune
        )
        self.text_encoder = TextEncoder(
            model_name=config.text_model,
            pretrained=config.use_pretrained,
            trainable=config.fine_tune
        )
        self.image_projection = ProjectionHead(embedding_dim=config.image_embedding_size)
        self.text_projection = ProjectionHead(embedding_dim=config.text_embedding_size)
        self.temperature = nn.Parameter(torch.tensor(config.temperature))

    def forward(self, image, input_ids, attention_mask):
        # Encode images and texts
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Project to shared embedding space
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Normalize the embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Compute logits using dot product and temperature
        logits = (text_embeddings @ image_embeddings.T) / self.temperature

        # Calculate the contrastive loss
        targets = torch.arange(len(image_embeddings)).to(image_embeddings.device)
        image_loss = F.cross_entropy(logits, targets)
        text_loss = F.cross_entropy(logits.T, targets)
        loss = (image_loss + text_loss) / 2
        return loss, image_embeddings, text_embeddings
