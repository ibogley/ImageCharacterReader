from collections import Counter
import numpy as np
import torch
import pandas as pd

model = torch.load('CharacterClassification/Models/Model1/Model1.pth')

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
