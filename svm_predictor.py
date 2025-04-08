"""
Title: Prediction of New Data Using Pre-Trained SVM Model

Author: Pablo García-Peña
Date: April 08, 2025

Description:
This script loads a pre-trained Support Vector Machine (SVM) model, scales new data, 
makes predictions, and stores the results. It is designed for use in predictive analytics, 
particularly in scenarios where SVM models are used to classify data into predefined categories 
based on feature inputs. The predictions are saved into a CSV file for further analysis.

Article Reference:
This code is part of the analysis presented in the article "[Title of the Article]" by [Authors],
published in [Journal Name], [Year]. Please cite this article when using or adapting this code.

License:
This work is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0).
You are free to share and adapt the material as long as appropriate credit is given.
For details, visit: https://creativecommons.org/licenses/by/4.0/

Dependencies:
- pandas==2.2.3
- numpy==2.2.1
- joblib==1.4.2
- scikit-learn==1.6.0
- matplotlib==3.10.0
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the pre-trained best SVM model
model = joblib.load('best_svm_model.pkl')
scaler = joblib.load('svm_scaler.pkl')

# Load the new data for making predictions
new_data = pd.read_excel("new_data_for_prediction.xlsx")

# Extract features for prediction
feature_names = scaler.feature_names_in_
X_new = new_data[feature_names]

# Scale the features using the pre-trained scaler
X_new = scaler.transform(X_new)

# Make predictions using the pre-trained model
y_pred = model.predict(X_new)

# Map numerical predictions to textual labels
label_mapping = {0: 'WT', 1: '5XFAD'}
predicted_labels_text = [label_mapping[label] for label in y_pred]

# Display all predictions with their corresponding row (prediction number)
for i, prediction in enumerate(predicted_labels_text):
    print(f"Prediction {i+1}: {prediction}")

# Save the results to an Excel file (check 'Prediction' column with the predicted labels)

new_data['Prediction'] = predicted_labels_text
new_data.to_excel('predictions.xlsx', index=False)
if len(predicted_labels_text) == 1:
    print("\nPrediction completed.")
    print("Prediction saved to 'predictions.xlsx'")
else:
    print("\nPredictions completed.")
    print("Predictions saved to 'predictions.xlsx'")