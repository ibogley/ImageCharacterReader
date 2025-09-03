import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


#Goal: Have a dataset class for each type of preprocessing we want to do that can be called easily in any script
class DatasetGrayscale:
    """Pass the dataset as a class to ensure all models work with the same dataset"""
    def __init__(self,train_portion = .8, seed=42,data_dir = None):
        # If no data directory is provided, use the last one in the 'data' folder
        if data_dir is None:
            data_dir = 'CharacterClassification/data/'+os.listdir('CharacterClassification/data')[-1]

        #Preprocess data to grayscale
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        #Get full dataset
        full_dataset = datasets.ImageFolder(root=data_dir, transform=self.transform)

        #Get the classes
        self.classes = full_dataset.classes
        #Get the number of classes
        self.num_classes = len(full_dataset.classes)

        # Split dataset
        train_size = int(len(full_dataset) * train_portion)
        test_size = int(len(full_dataset))-train_size
        
        self.train_dataset, self.test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
        
        #Get Dataloaders
        self.train_loader = DataLoader(self.train_dataset, num_workers=12, pin_memory=True, batch_size=512, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, num_workers=12, pin_memory=True, batch_size=512, shuffle=False)
