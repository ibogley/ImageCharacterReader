from tqdm import tqdm
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import AnalysisFunctions as AF
from Preprocessing import DatasetGrayscale
import torch.nn as nn
from Models.Model2.Training import Model2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load the model
model = Model2(num_classes=94)
model.load_state_dict(torch.load('Models/Model2/model2.pth'))
model = model.to(device)


DataObject = DatasetGrayscale()
train_loader = DataObject.train_loader
test_loader = DataObject.test_loader

#------------------------------------------------------
#Final predictions for testing dataset, compare accuracy
#------------------------------------------------------
predictions_raw = AF.get_predictions(model, test_loader, device,class_names=DataObject.classes)
predictions = predictions_raw['Full Dataframe']
#Accuracy of 98.42%
print(str(round(predictions['AccuratePrediction'].mean()*100,4))+'% accuracy on test set')
class_accuracies = predictions.groupby('True').mean('AccuratePrediction').sort_values('AccuratePrediction')['AccuratePrediction']

#For each class, get the class it was most confused with
confusion_matrix = pd.crosstab(predictions['True'], predictions['Predicted'], rownames=['Actual'], colnames=['Predicted'], normalize='index')
most_confused = confusion_matrix.apply(lambda x: x.nlargest(2).idxmin(), axis=1)
confusion_summary = pd.DataFrame({'Accuracy': class_accuracies, 'MostConfusedWith': most_confused}).sort_values('Accuracy')
confusion_summary.head(10)

AF.show_incorrect_predictions(predictions, test_loader, 'l_lower', num_images=10)
AF.show_incorrect_predictions(predictions, test_loader, 'F_upper', num_images=10)
AF.show_incorrect_predictions(predictions, test_loader, '1_digit', num_images=10)
AF.show_incorrect_predictions(predictions, test_loader, 'RightParenthesis_punct', num_images=10)