# Model Creation Documentation: classifier_bal_1

This document details the creation process of the `classifier_bal_1` model, which predicts user rewards based on activity patterns in the Karma platform.

## Overview

The model is a Random Forest Classifier that predicts whether a user should receive a reward based on their activity metrics. It's trained on synthetic data that simulates user behavior and reward distribution.

## Data Preparation

### Input Data
- Training data: `training_data_bal_1.json`
- Validation data: `validation_data_bal_1.json`
- Testing data: `testing_data_bal_1.json`

### Data Loading
1. **Dataset Loading**:
   - Loads JSON files containing user activity data
   - Extracts features and labels
   - Each sample includes user metrics like login streak, posts created, comments written, etc.

2. **Conditions Loading**:
   - Loads reward conditions from `conditions.csv`
   - Parses condition strings into executable expressions
   - Filters out invalid conditions

## Feature Engineering

### 1. Rule-based Features
- Creates additional features based on the reward conditions
- Each condition is evaluated as a binary feature (0 or 1)
- Helps the model learn the underlying reward rules

### 2. Temporal Features
- Extracts day of week and month from timestamps
- Applies temporal multipliers based on the day and season
- Captures patterns like weekend vs weekday activity

### 3. User Clustering
- Assigns users to behavioral clusters (Casual, Social, ContentCreator, etc.)
- Based on cluster probabilities defined in `config.json`
- Helps capture different user behavior patterns

## Data Balancing

### SMOTE-R (Synthetic Minority Over-sampling Technique - Rule-based)
- Applies SMOTE to handle class imbalance
- Special handling to ensure synthetic samples satisfy at least one reward condition
- Uses `k_neighbors=3` for generating synthetic samples
- Only applied to the training data to prevent data leakage

## Model Architecture

### Base Model
- **Algorithm**: Random Forest Classifier
- **Key Parameters**:
  - `n_estimators`: 300
  - `max_depth`: 100
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
  - `random_state`: 42
  - `n_jobs`: -1 (uses all available cores)

### Model Training
1. **Cross-Validation**:
   - 5-fold cross-validation
   - Stratified to maintain class distribution
   - Shuffled with fixed random state for reproducibility

2. **Hyperparameter Tuning**:(Note: I have used the previous results to find the optimal hyperparameters)
   - Grid search over parameter space
   - Evaluated using accuracy metric
   - Parameters searched:
     - `n_estimators`: [200]
     - `max_depth`: [50]
     - `min_samples_split`: [5]
     - `min_samples_leaf`: [5]

3. **Optimal Threshold Selection**:
   - Finds the probability threshold that maximizes accuracy on validation data
   - Applied during inference to convert probabilities to class labels

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve

### Cross-Validation Results
My previous results
{
  "cross_validation_metrics": {
    "accuracy": [
      0.9152157829839704,
      0.9094944512946979,
      0.9140813810110974,
      0.9120548485745289,
      0.9125480911512281
    ],
    "precision": [
      0.8995166387804424,
      0.8907214595042845,
      0.8965071151358344,
      0.888295423151303,
      0.8978984563883207
    ],
    "recall": [
      0.9381483276781386,
      0.9370880186118651,
      0.9397520340953119,
      0.9442051683633516,
      0.9346626657632369
    ],
    "f1": [
      0.9184264224362929,
      0.9133166422599084,
      0.9176203537312021,
      0.9153973902728351,
      0.9159117856295945
    ],
    "auc_roc": [
      0.9731630740119955,
      0.9710754233886659,
      0.9728249155071397,
      0.9726180362199944,
      0.9729923539686814
    ]
  },
  "average_metrics": {
    "accuracy": 0.9126789110031046,
    "precision": 0.8945878185920371,
    "recall": 0.9387712429023807,
    "f1": 0.9161345188659664,
    "auc_roc": 0.9725347606192953
  },
  "test_metrics": {
    "accuracy": 0.9076318934608161,
    "precision": 0.9060841663581389,
    "recall": 0.8949293027791322,
    "f1": 0.9004721898571165,
    "auc_roc": 0.9744703713884605
  },
  "timestamp": "2025-05-29 17:39:49.271574"
}
## Model Persistence

The final model is saved as:
- `classifier_bal_1.pkl`: The trained Random Forest model

## Usage

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load("classifier_bal_1.pkl")

# Prepare input data (should match training features)
# ...

# Make predictions
predictions = model.predict_proba(input_data)[:, 1]  # Get probability scores
binary_predictions = (predictions >= optimal_threshold).astype(int)  # Apply threshold
```

## Dependencies

- Python 3.7+
- scikit-learn
- pandas
- numpy
- imbalanced-learn (for SMOTE)
- joblib (for model persistence)

## Training Configuration

### Hardware
- Multi-core CPU recommended
- Uses all available CPU cores for training (`n_jobs=-1`)

### Memory
- Minimum: 8GB RAM
- Recommended: 16GB+ RAM for larger datasets

## Model Monitoring and Maintenance

### Performance Monitoring
- Monitor metrics on validation set over time
- Track distribution shifts in input features
- Watch for degradation in precision/recall