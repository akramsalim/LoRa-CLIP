# clip/config.py

import torch

class Config:
    """
    Configuration class for managing hyperparameters, paths, and settings 
    used throughout the CLIP training and inference process.
    """
    
    def __init__(self, dataset_name="oxford_pets"):
        # Dataset settings
        self.dataset_name = dataset_name
        self.data_dir = "./data"  # Base directory for storing dataset
        self.image_dir = f"{self.data_dir}/{self.dataset_name}/images"
        self.annotation_file = f"{self.data_dir}/captions.csv"

        # Model settings
        self.image_model = 'resnet50'
        self.text_model = "distilbert-base-uncased"
        self.use_pretrained = True  # Use pre-trained weights for both image and text models
        self.fine_tune = True       # Allow training of pre-trained layers

        # Embedding settings
        self.image_embedding_size = 2048
        self.text_embedding_size = 768
        self.projection_size = 256  # Dimensionality of the projection space
        self.max_text_length = 77   # Maximum number of tokens for text input

        # Training hyperparameters
        self.batch_size = 64
        self.num_workers = 4
        self.epochs = 20
        self.learning_rate = {
            "image_encoder": 1e-4,
            "text_encoder": 1e-5,
            "projection_head": 1e-3
        }
        self.weight_decay = 1e-3
        self.scheduler_patience = 3
        self.scheduler_factor = 0.5

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training behavior settings
        self.gradient_accumulation_steps = 1
        self.temperature = 0.07  # Temperature parameter for the contrastive loss

        # Image preprocessing settings
        self.input_image_size = 224  # Image size for resizing during preprocessing
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # Paths for saving models and logs
        self.model_save_path = "./models"
        self.best_model_file = f"{self.model_save_path}/clip_best_model.pth"
        self.logs_dir = "./logs"

    def update_learning_rate(self, new_rates):
        """
        Update learning rates for the model components.
        :param new_rates: dict with keys "image_encoder", "text_encoder", and "projection_head"
        """
        self.learning_rate.update(new_rates)

    def display(self):
        """
        Print out the current configuration settings.
        """
        print("Configuration Settings:")
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
