from tqdm import tqdm
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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