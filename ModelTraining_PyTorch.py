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

torch.backends.cudnn.benchmark = True

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
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, num_workers = 8,pin_memory=True,batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  num_workers = 8,pin_memory=True,batch_size=batch_size, shuffle=False)

# ----------------------------
# 4. Define a basic CNN model
# ----------------------------
num_classes = 94  # Adjust based on your dataset
class CNN94(nn.Module):
    def __init__(self, num_classes=94):
        super(CNN94, self).__init__()
        
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
model = CNN94(num_classes=num_classes).to(device)

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
    # TESTING PHASE
    # ----------------------------
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = 100 * test_correct / test_total

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

from collections import Counter

def get_class_distribution_fast(dataset):
    """
    Get class distribution efficiently using NumPy.
    Works with ImageFolder and Subset datasets.
    """
    # If dataset is a Subset, unwrap the original dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    # Make sure dataset is from torchvision.datasets.ImageFolder
    if hasattr(dataset, "targets"):  
        # Convert targets to NumPy array for vectorization
        labels = np.array(dataset.targets)

        # Count unique labels
        counts = np.bincount(labels)

        # Map class indices to class names
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(counts))]

        # Create a sorted DataFrame
        df = pd.DataFrame({"Class": class_names, "Count": counts})
        df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)
        return df
    else:
        raise AttributeError(
            "Dataset does not have 'targets'. Make sure you're using ImageFolder or similar."
        )


# Training and testing distributions
train_dist = get_class_distribution_fast(train_loader.dataset)
test_dist = get_class_distribution_fast(test_loader.dataset)

print("\nTraining Dataset Class Distribution:")
print(train_dist)
print("\nTesting Dataset Class Distribution:")
print(test_dist)

#------------------------------------------------------
#Final predictions for testing dataset, compare accuracy
#------------------------------------------------------

def get_per_class_accuracy(model, test_loader, device):
    """
    Evaluate per-class accuracy on the test dataset.
    Returns a DataFrame with counts, correct predictions, and accuracy.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            # Get predictions
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Collect labels and predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to NumPy arrays for vectorized operations
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Count total samples per class
    unique_classes, total_counts = np.unique(all_labels, return_counts=True)

    # Count correct predictions per class
    correct_counts = np.array([
        np.sum((all_preds == all_labels) & (all_labels == cls))
        for cls in unique_classes
    ])

    # Calculate per-class accuracy
    accuracy = correct_counts / total_counts

    # Map class indices to class names
    if isinstance(test_loader.dataset, torch.utils.data.Subset):
        dataset = test_loader.dataset.dataset
    else:
        dataset = test_loader.dataset

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in unique_classes]

    # Create a DataFrame
    df = pd.DataFrame({
        "Class": class_names,
        "Total Samples": total_counts,
        "Correct Predictions": correct_counts,
        "Accuracy (%)": np.round(accuracy * 100, 2)
    })

    df = df.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
    return df

def show_mistakes_for_class(model, test_loader, target_class, n=9, device=None):
    """
    Show the first n mistakes made by the model for a given class.
    """
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_images = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_preds.extend(preds.cpu())

    all_images = torch.stack(all_images)
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)

    # Map indices to class names
    if isinstance(test_loader.dataset, torch.utils.data.Subset):
        dataset = test_loader.dataset.dataset
    else:
        dataset = test_loader.dataset
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Get the index of the target class
    class_idx = dataset.class_to_idx[target_class]

    # Find misclassified images for the given class
    mistakes_idx = torch.where((all_labels == class_idx) & (all_preds != all_labels))[0]

    if len(mistakes_idx) == 0:
        print(f"No mistakes found for class '{target_class}' ✅")
        return

    # Limit to first n mistakes
    mistakes_idx = mistakes_idx[:n]
    mistake_images = all_images[mistakes_idx]
    true_labels = all_labels[mistakes_idx]
    predicted_labels = all_preds[mistakes_idx]

    # Plotting grid
    rows = int(np.ceil(n / 3))
    plt.figure(figsize=(12, 4 * rows))

    for i, idx in enumerate(mistakes_idx):
        plt.subplot(rows, 3, i + 1)

        img = mistake_images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis("off")

        true_class = idx_to_class[int(true_labels[i])]
        pred_class = idx_to_class[int(predicted_labels[i])]
        plt.title(f"True: {true_class}\nPred: {pred_class}", color="red")

    plt.suptitle(f"First {n} Mistakes for Class '{target_class}'", fontsize=16)
    plt.tight_layout()
    plt.show()

per_class_df = get_per_class_accuracy(model, test_loader, device)

per_class_df.sort_values(by="Accuracy (%)", ascending=True, inplace=True)

show_mistakes_for_class(model, test_loader, target_class=per_class_df.iloc[0]["Class"], n=10, device=device)

with pd.option_context('display.max_rows', None):
        print(per_class_df)

#Final accuracy: 96.4%
sum(per_class_df['Correct Predictions']) / sum(per_class_df['Total Samples'])

#Worst issues with:
#I_upper (73%)
#0_digit (82%)
#l_lower (84%)
#o_lower (85%)
#All others over 90% accuracy
#Save the model
if 'models' not in os.listdir():
    os.makedirs('Models')
torch.save(model.state_dict(), 'Models/Model1.pth')

#Next steps:
#1. Seperate out training script and model analysis
#2. Change model analysis to include a simple df of predictions and true labels for each entry
#3. Identify issues with individual classes, as well as if there is anything I can do to combat
#4. I don't think we are overfitting yet. Look for opportunities to increase the dataset size or model capacity
#5. Experiment with different architectures or hyperparameters