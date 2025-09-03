import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Create object with predictions and probabilities of differenct classes
def get_predictions(model, data_loader, device, class_names=None):
    # Move model to the correct device
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            # Move images and labels to same device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)  # logits

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Get predicted classes
            _, predicted = torch.max(probs, 1)

            # Store results
            all_labels.append(labels)
            all_preds.append(predicted)
            all_probs.append(probs)

    # Concatenate results across batches
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy()

    # Create a dataframe of true vs predicted labels
    results_df = pd.DataFrame({'True': all_labels, 'Predicted': all_preds,'AccuratePrediction': all_labels==all_preds})

    # Create a dataframe of predicted probabilities
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(all_probs.shape[1])]
    probs_df = pd.DataFrame(all_probs, columns=class_names)

    #Unify predictions and probabilities
    full_df = pd.merge(results_df, probs_df, left_index=True, right_index=True)
    
    probs_array = probs_df.to_numpy()
    # Get column indices for true and predicted labels
    true_indices = full_df['True'].to_numpy()
    pred_indices = full_df['Predicted'].to_numpy()

    # Vectorized indexing using NumPy
    full_df['TrueProbability'] = probs_array[np.arange(len(full_df)), true_indices]
    full_df['PredictedProbability'] = probs_array[np.arange(len(full_df)), pred_indices]

    #If provided class names, map the numeric labels to class names
    if class_names is not None:
        full_df['True'] = full_df['True'].map(lambda x: class_names[x])
        full_df['Predicted'] = full_df['Predicted'].map(lambda x: class_names[x])

    return {'TrueLabels': all_labels,'Predictions': all_preds,'Full Dataframe': full_df}

#Take a class and return the first 10 incorrect predictions
def show_incorrect_predictions(predictions_df,data_loader,class_name,num_images=10):
    incorrect_preds = predictions_df[(predictions_df['True'] == class_name) & (predictions_df['AccuratePrediction'] == False)]
    incorrect_preds_indexes = incorrect_preds.index.tolist()

    if len(incorrect_preds_indexes) == 0:
        print(f"No incorrect predictions found for class {class_name}.")
        return None
    
    if len(incorrect_preds_indexes) < num_images:
        num_images = len(incorrect_preds_indexes)
        print(f"Only found {num_images} incorrect predictions for class {class_name}. Displaying all.")

    plt.subplots(np.ceil(num_images/5).astype(int), 5, figsize=(15, 6))
    for i, idx in enumerate(incorrect_preds_indexes[:num_images]):
        img, label = data_loader.dataset[idx]
        img = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for plotting
        plt.subplot(2, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {predictions_df.loc[idx, 'True']}\nPred: {predictions_df.loc[idx, 'Predicted']}\nProb: {predictions_df.loc[idx, 'PredictedProbability']:.2f}")
        plt.axis('off')
        plt.tight_layout()
    plt.show()
    return None