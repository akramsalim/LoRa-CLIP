# clip/text_encoder.py

import torch.nn as nn
from transformers import DistilBertModel
from .config import Config

class TextEncoder(nn.Module):
    """
    Encodes text into fixed-size vectors using a pre-trained DistilBERT model.
    """
    def __init__(self, model_name=Config().text_model, pretrained=True, trainable=True):
        super(TextEncoder, self).__init__()
        self.model = DistilBertModel.from_pretrained(model_name) if pretrained else DistilBertModel()
        for param in self.model.parameters():
            param.requires_grad = trainable
        self.target_token_idx = 0  # Use the [CLS] token hidden state as the sentence embedding

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, self.target_token_idx, :]
