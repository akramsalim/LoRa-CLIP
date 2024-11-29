# clip/image_encoder.py

import torch.nn as nn
import timm
from .config import Config

class ImageEncoder(nn.Module):
    """
    Encodes images into fixed-size vectors using a pre-trained vision model.
    """
    def __init__(self, model_name=Config().image_model, pretrained=True, trainable=True):
        super(ImageEncoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        # Freeze or unfreeze the layers based on trainable setting
        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
