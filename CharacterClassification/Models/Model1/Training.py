import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import datetime
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np

#torch.backends.cudnn.benchmark = True

# ----------------------------
# 1. Device configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Hyperparameters
# ----------------------------
batch_size = 256
learning_rate = 0.001
num_epochs = 20

# ----------------------------
# 3. Data loading & transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = 'data/'+os.listdir('data')[-1]

full_dataset = datasets.ImageFolder(root = data_dir,transform=transform)


# Calculate train/test split sizes
test_split = .8
total_size = len(full_dataset)
test_size = int(total_size * test_split)
train_size = total_size - test_size

# Split dataset
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

# DataLoaders
train_loader = DataLoader(train_dataset, num_workers = 8,pin_memory=True,batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  num_workers = 8,pin_memory=True,batch_size=batch_size, shuffle=False)

# ----------------------------
# 4. Define a basic CNN model
# ----------------------------
num_classes = 94  # Adjust based on your dataset
class Model1(nn.Module):
    def __init__(self, num_classes=94):
        super(Model1, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 12 * 12, 256)  # Adjusted for 100x100 input after 3 pools
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv blocks: Conv -> ReLU -> Pool -> Dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize model
model = Model1(num_classes=num_classes).to(device)

# ----------------------------
# 5. Loss and optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# 6. Training loop
# ----------------------------
starttime = datetime.datetime.now()
history = []  # Will store results for each epoch

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Training phase with tqdm progress bar
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss & accuracy
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Live update for current batch
        train_acc = 100 * correct / total
        train_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Train Acc": f"{train_acc:.2f}%"})

    # Epoch-level training metrics
    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total

    # ----------------------------
    # EVALUATE TESTING ACCURACY
    # ----------------------------
    """model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = 100 * test_correct / test_total"""
    
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and labels
            all_preds.append(preds)
            all_labels.append(labels)

    # Concatenate tensors instead of Python summing
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Vectorized accuracy computation
    test_acc = (all_preds == all_labels).float().mean().item() * 100


    # Print summary for this epoch
    print(f"Timestamp: [{datetime.datetime.now()}] "
          f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% " 
          f"Test Acc: {test_acc:.2f}%")

    # Log metrics for visualization later
    history.append({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Train Acc": train_acc,
        "Test Acc": test_acc
    })

endtime = datetime.datetime.now()
print(f"Training complete in: {endtime - starttime}")

plt.plot([x['Epoch'] for x in history], [x['Train Acc'] for x in history], label='Train Acc')
plt.plot([x['Epoch'] for x in history], [x['Test Acc'] for x in history], label='Test Acc  ')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()


