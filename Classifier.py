import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load training and testing datasets
with open('training_data.json', 'r') as f:
    training_data = json.load(f)

with open('testing_data.json', 'r') as f:
    testing_data = json.load(f)

# Extract features and labels
X_train = pd.DataFrame([data_point['features'] for data_point in training_data])
y_train = np.array([data_point['label'] for data_point in training_data])
X_test = pd.DataFrame([data_point['features'] for data_point in testing_data])
y_test = np.array([data_point['label'] for data_point in testing_data])

# Print data shapes and label distribution
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("\nTraining Label Distribution:")
print(pd.Series(y_train).value_counts())

# Handle class imbalance with sample weights
weights = compute_sample_weight('balanced', y_train)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest Classifier
rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',  # Optimize for balanced performance
    n_jobs=-1,
    verbose=1
)

# Train the model with sample weights
grid_search.fit(X_train, y_train, sample_weight=weights)

# Get the best model
best_rf = grid_search.best_estimator_
print("\nBest Hyperparameters:", grid_search.best_params_)

# Make predictions on the testing data
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Evaluate performance with multiple metrics
print("\nModel Performance on Testing Data:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Feature Importance Analysis
feature_importances = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df)

# Visualize Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

# Probability Threshold Tuning
thresholds = [0.4, 0.5, 0.6, 0.7]
print("\nPerformance at Different Probability Thresholds:")
for threshold in thresholds:
    y_pred_adj = (y_pred_proba >= threshold).astype(int)
    print(f"\nThreshold = {threshold}:")
    print(classification_report(y_test, y_pred_adj))

# Save the best model and feature names
joblib.dump(best_rf, 'classifier.pkl')
feature_names = X_train.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

print("\nBest model saved as 'classifier.pkl' and feature names saved as 'feature_names.json'.")
##
#(venv) PS D:\Karma-AI> python Classifier.py
# Training data shape: (5100, 8)
# Testing data shape: (1020, 8)

# Training Label Distribution:
# 1    4600
# 0     500
# Name: count, dtype: int64
# Fitting 5 folds for each of 108 candidates, totalling 540 fits

# Best Hyperparameters: {'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 300}

# Model Performance on Testing Data:
# Accuracy: 0.957843137254902

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.70      1.00      0.82       100
#            1       1.00      0.95      0.98       920

#     accuracy                           0.96      1020
#    macro avg       0.85      0.98      0.90      1020
# weighted avg       0.97      0.96      0.96      1020


# Feature Importances:
#               Feature  Importance
# 3    upvotes_received    0.431713
# 1       posts_created    0.276560
# 7  karma_earned_today    0.129450
# 2    comments_written    0.087945
# 0        login_streak    0.029388
# 6         karma_spent    0.017173
# 5    buddies_messaged    0.014385
# 4   quizzes_completed    0.013386

# Performance at Different Probability Thresholds:

# Threshold = 0.4:
#               precision    recall  f1-score   support

#            0       0.76      1.00      0.86       100
#            1       1.00      0.97      0.98       920

#     accuracy                           0.97      1020
#    macro avg       0.88      0.98      0.92      1020
# weighted avg       0.98      0.97      0.97      1020


# Threshold = 0.5:
#               precision    recall  f1-score   support

#            0       0.70      1.00      0.82       100
#            1       1.00      0.95      0.98       920

#     accuracy                           0.96      1020
#    macro avg       0.85      0.98      0.90      1020
# weighted avg       0.97      0.96      0.96      1020


# Threshold = 0.6:
#               precision    recall  f1-score   support

#            0       0.68      1.00      0.81       100
#            1       1.00      0.95      0.97       920

#     accuracy                           0.95      1020
#    macro avg       0.84      0.97      0.89      1020
# weighted avg       0.97      0.95      0.96      1020


# Threshold = 0.7:
#               precision    recall  f1-score   support

#            0       0.49      1.00      0.66       100
#            1       1.00      0.89      0.94       920

#     accuracy                           0.90      1020
#    macro avg       0.75      0.94      0.80      1020
# weighted avg       0.95      0.90      0.91      1020












# # Import necessary libraries
# import json
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import joblib

# # Load the training dataset from JSON
# with open('training_data.json', 'r') as f:
#     training_data = json.load(f)

# # Load the testing dataset from JSON
# with open('testing_data.json', 'r') as f:
#     testing_data = json.load(f)

# # Extract features and labels for training data
# # Features are stored as dictionaries in each data point; convert to a DataFrame
# X_train = pd.DataFrame([data_point['features'] for data_point in training_data])
# # Labels are binary (0 or 1); convert to a NumPy array
# y_train = np.array([data_point['label'] for data_point in training_data])

# # Extract features and labels for testing data
# X_test = pd.DataFrame([data_point['features'] for data_point in testing_data])
# y_test = np.array([data_point['label'] for data_point in testing_data])

# # Display the distribution of labels in the training data to check for balance
# print("Training Label Distribution:")
# print(pd.Series(y_train).value_counts())

# # Initialize the Random Forest Classifier
# # n_estimators=100 sets the number of trees; random_state=42 ensures reproducibility
# rf = RandomForestClassifier(n_estimators=100, random_state=42)

# # Train the model using the training data
# rf.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = rf.predict(X_test)

# # Evaluate the model's performance
# print("\nModel Performance on Testing Data:")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # Save the trained model to a file for later use in the microservice
# joblib.dump(rf, 'classifier.pkl')

# # Save the feature names to ensure consistent input ordering in the microservice
# feature_names = X_train.columns.tolist()
# with open('feature_names.json', 'w') as f:
#     json.dump(feature_names, f)

# print("\nTrained model saved as 'classifier.pkl' and feature names saved as 'feature_names.json'.")
# Training Label Distribution:
# 1    4600
# 0     500
# Name: count, dtype: int64

# Model Performance on Testing Data:
# Accuracy: 0.9715686274509804

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.86      0.85      0.85       100
#            1       0.98      0.98      0.98       920


#            0       0.86      0.85      0.85       100
#            1       0.98      0.98      0.98       920

#            0       0.86      0.85      0.85       100
#            1       0.98      0.98      0.98       920


#     accuracy                           0.97      1020
#     accuracy                           0.97      1020
#    macro avg       0.92      0.92      0.92      1020
# weighted avg       0.97      0.97      0.97      1020
# weighted avg       0.97      0.97      0.97      1020


# Confusion Matrix:
# [[ 85  15]
#  [ 14 906]]

# Trained model saved as 'classifier.pkl' and feature names saved as 'feature_names.json'.