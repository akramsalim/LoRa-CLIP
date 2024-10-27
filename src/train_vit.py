import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from vision_transformer import VisionTransformer

# Hyperparameters
img_size = 224  # Resize images to 224x224
patch_size = 16  # Size of each patch
num_classes = 102  # Caltech101 has 101 classes + 1 background class
d_model = 128  # Dimensionality of the model
num_heads = 8  # Number of attention heads
num_layers = 6  # Number of transformer encoder layers
d_ff = 512  # Dimension of feed-forward layers
dropout = 0.1  # Dropout rate
batch_size = 64  # Batch size per gradient accumulation step
accumulation_steps = 4  # Accumulate gradients over 4 steps to simulate a larger batch size
num_epochs = 100  # Total number of training epochs
learning_rate = 6e-4  # Learning rate for the optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure all images have 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the full Caltech101 dataset
full_dataset = Caltech101(root='./data', download=True, transform=transform)

# Define the train-validation split
train_size = int(0.8 * len(full_dataset))  # 80% for training
val_size = len(full_dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Initialize the Vision Transformer model
model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    num_classes=num_classes,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    dropout=dropout
).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Initialize the gradient scaler for mixed precision
scaler = GradScaler()

# Training function with gradient accumulation and mixed precision
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    optimizer.zero_grad()  # Zero the gradients at the start of each epoch

    for step, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Use autocast for mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation

        # Scale the loss and perform backward pass
        scaler.scale(loss).backward()

        # Perform weight update after accumulating gradients for 'accumulation_steps' batches
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Reset gradients after updating

        running_loss += loss.item() * images.size(0) * accumulation_steps  # Rescale for logging
        _, preds = outputs.max(1)
        correct_predictions += preds.eq(labels).sum()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct_predictions += preds.eq(labels).sum()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

# Training loop
best_val_acc = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the model if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'vit_caltech101_best.pth')
        print("Saved Best Model!")

    # Adjust the learning rate
    scheduler.step()

# Save the final model
torch.save(model.state_dict(), 'vit_caltech101_final.pth')
print("Training complete!")
