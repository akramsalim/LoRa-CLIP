# clip/dataset.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from torchvision.datasets import OxfordIIITPet, Caltech101
from transformers import DistilBertTokenizer
#from .config import Config
from clip.config import Config

class CLIPDataset(Dataset):
    """
    Custom Dataset class for loading images and captions for CLIP.
    Supports image transformations and text tokenization.
    """
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        self.image_filenames = image_filenames
        self.captions = captions
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=Config().max_text_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}

        # Load and preprocess the image
        image_path = self.image_filenames[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()

        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)

def get_transforms():
    """
    Returns the transformation pipeline for training and validation images.
    """
    return A.Compose([
        A.Resize(Config().input_image_size, Config().input_image_size, always_apply=True),
        A.Normalize(mean=Config().image_normalization_mean, std=Config().image_normalization_std, always_apply=True),
    ])

def load_dataset():
    """
    Loads the specified dataset, creates training and validation splits, and returns DataLoader objects.
    """
    config = Config()
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_model)
    
    # Choose the dataset based on the config
    if config.dataset_name == "oxford_pets":
        dataset = OxfordIIITPet(root=config.data_dir, download=True)
        captions = [f"{item[1]}" for item in dataset]  # Using labels as captions for simplicity
        image_filenames = [item[0] for item in dataset]
    elif config.dataset_name == "caltech101":
        dataset = Caltech101(root=config.data_dir, download=True)
        captions = [f"{item[1]}" for item in dataset]
        image_filenames = [item[0] for item in dataset]
    else:
        raise ValueError(f"Dataset {config.dataset_name} is not supported")

    # Create train-validation split
    num_samples = len(image_filenames)
    indices = torch.randperm(num_samples).tolist()
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    # Create datasets
    train_dataset = CLIPDataset(
        [image_filenames[i] for i in train_indices],
        [captions[i] for i in train_indices],
        tokenizer=tokenizer,
        transforms=get_transforms()
    )
    val_dataset = CLIPDataset(
        [image_filenames[i] for i in val_indices],
        [captions[i] for i in val_indices],
        tokenizer=tokenizer,
        transforms=get_transforms()
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader
