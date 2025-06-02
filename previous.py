# import json
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import shap
# import joblib
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Define expected features (aligned with dataset.py)
# MODEL_FEATURE_KEYS = ["login_streak", "posts_created", "comments_written", "upvotes_received", "quizzes_completed", "buddies_messaged", "karma_spent", "karma_earned_today"]

# # Create a local RNG to avoid FutureWarning
# rng = np.random.default_rng(42)

# # Load datasets
# def load_dataset(filename):
#     with open(filename, 'r') as f:
#         data = json.load(f)
#     X = pd.DataFrame([s["features"] for s in data])
#     y = np.array([s["label"] for s in data])
    
#     # Ensure only expected features are included and in the correct order
#     missing_features = [feat for feat in MODEL_FEATURE_KEYS if feat not in X.columns]
#     if missing_features:
#         raise ValueError(f"Missing features in {filename}: {missing_features}")
#     extra_features = [feat for feat in X.columns if feat not in MODEL_FEATURE_KEYS]
#     if extra_features:
#         print(f"Warning: Extra features in {filename}: {extra_features}. Dropping them.")
#         X = X.drop(columns=extra_features)
#     X = X[MODEL_FEATURE_KEYS]  # Reorder columns to match MODEL_FEATURE_KEYS
#     return X, y

# print("Loading datasets...")
# X_train, y_train = load_dataset("training_data.json")
# X_val, y_val = load_dataset("validation_data.json")
# X_test, y_test = load_dataset("testing_data.json")

# # Combine training and validation for cross-validation
# X_train_val = pd.concat([X_train, X_val], axis=0)
# y_train_val = np.concatenate([y_train, y_val])

# # Debug: Print shapes and columns
# print(f"X_train_val shape: {X_train_val.shape}")
# print(f"X_train_val columns: {list(X_train_val.columns)}")
# print(f"X_test shape: {X_test.shape}")
# print(f"X_test columns: {list(X_test.columns)}")

# # Define K-Fold cross-validation
# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Initialize lists to store metrics for each fold
# fold_metrics = {
#     "accuracy": [],
#     "precision": [],
#     "recall": [],
#     "f1": [],
#     "auc_roc": []
# }

# # Initialize model
# model = RandomForestClassifier(
#     n_estimators=150,
#     max_depth=10,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     random_state=42,
#     n_jobs=-1
# )

# # Perform K-Fold cross-validation
# print(f"Starting {n_splits}-fold cross-validation...")
# fold_num = 1
# for train_idx, val_idx in kf.split(X_train_val):
#     print(f"\nFold {fold_num}/{n_splits}")
    
#     # Split data
#     X_fold_train, X_fold_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
#     y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
    
#     # Train model
#     model.fit(X_fold_train, y_fold_train)
    
#     # Predict on validation fold
#     y_pred = model.predict(X_fold_val)
#     y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    
#     # Calculate metrics
#     accuracy = accuracy_score(y_fold_val, y_pred)
#     precision = precision_score(y_fold_val, y_pred)
#     recall = recall_score(y_fold_val, y_pred)
#     f1 = f1_score(y_fold_val, y_pred)
#     auc_roc = roc_auc_score(y_fold_val, y_pred_proba)
    
#     # Store metrics
#     fold_metrics["accuracy"].append(accuracy)
#     fold_metrics["precision"].append(precision)
#     fold_metrics["recall"].append(recall)
#     fold_metrics["f1"].append(f1)
#     fold_metrics["auc_roc"].append(auc_roc)
    
#     # Print metrics for this fold
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1-Score: {f1:.4f}")
#     print(f"AUC-ROC: {auc_roc:.4f}")
    
#     fold_num += 1

# # Calculate and print average metrics across folds
# print("\n--- Average Metrics Across Folds ---")
# for metric, values in fold_metrics.items():
#     avg = np.mean(values)
#     std = np.std(values)
#     print(f"{metric.capitalize()}: {avg:.4f} ± {std:.4f}")

# # Train final model on combined training + validation data
# print("\nTraining final model on combined training + validation data...")
# model.fit(X_train_val, y_train_val)

# # Evaluate on test set
# print("\nEvaluating on test set...")
# y_test_pred = model.predict(X_test)
# y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# test_metrics = {
#     "accuracy": accuracy_score(y_test, y_test_pred),
#     "precision": precision_score(y_test, y_test_pred),
#     "recall": recall_score(y_test, y_test_pred),
#     "f1": f1_score(y_test, y_test_pred),
#     "auc_roc": roc_auc_score(y_test, y_test_pred_proba)
# }

# print("\n--- Test Set Metrics ---")
# for metric, value in test_metrics.items():
#     print(f"{metric.capitalize()}: {value:.4f}")

# # SHAP Explainer for feature importance
# print("\nComputing SHAP values for model interpretability...")
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)

# # Debug: Print shapes of shap_values and X_test
# print(f"shap_values[1] shape: {np.array(shap_values[1]).shape}")
# print(f"X_test shape: {X_test.shape}")

# # Plot SHAP summary (feature importance)
# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False, random_state=rng)
# plt.title("Feature Importance (SHAP Values) on Test Set")
# plt.tight_layout()
# plt.savefig("shap_feature_importance.png")
# plt.close()

# # Plot SHAP summary (detailed impact)
# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values[1], X_test, show=False, random_state=rng)
# plt.title("SHAP Summary Plot on Test Set")
# plt.tight_layout()
# plt.savefig("shap_summary_plot.png")
# plt.close()

# # Save the model
# print("\nSaving the trained model...")
# joblib.dump(model, "classifier.pkl")
# print("Model saved as 'classifier.pkl'")

# # Save metrics to a file
# metrics_report = {
#     "cross_validation_metrics": fold_metrics,
#     "average_metrics": {metric: np.mean(values) for metric, values in fold_metrics.items()},
#     "test_metrics": test_metrics,
#     "timestamp": str(datetime.now())
# }

# with open("training_metrics.json", "w") as f:
#     json.dump(metrics_report, f, indent=2)
# print("Metrics saved to 'training_metrics.json'")



# (venv) PS D:\Karma-AI> python Classifier.py 
# Loading datasets...
# X_train_val shape: (51196, 8)
# X_train_val columns: ['login_streak', 'posts_created', 'comments_written', 'upvotes_received', 'quizzes_completed', 'buddies_mes
# saged', 'karma_spent', 'karma_earned_today']                                                                                    X_test shape: (8941, 8)
# X_test columns: ['login_streak', 'posts_created', 'comments_written', 'upvotes_received', 'quizzes_completed', 'buddies_messaged
# ', 'karma_spent', 'karma_earned_today']                                                                                         Starting 5-fold cross-validation...

# Fold 1/5
# Accuracy: 0.9141
# Precision: 0.8945
# Recall: 0.9439
# F1-Score: 0.9185
# AUC-ROC: 0.9723

# Fold 2/5
# Accuracy: 0.9160
# Precision: 0.8964
# Recall: 0.9464
# F1-Score: 0.9207
# AUC-ROC: 0.9738

# Fold 3/5
# Accuracy: 0.9171
# Precision: 0.8974
# Recall: 0.9473
# F1-Score: 0.9217
# AUC-ROC: 0.9717

# Fold 4/5
# Accuracy: 0.9116
# Precision: 0.8910
# Recall: 0.9447
# F1-Score: 0.9171
# AUC-ROC: 0.9710

# Fold 5/5
# Accuracy: 0.9117
# Precision: 0.8938
# Recall: 0.9429
# F1-Score: 0.9177
# AUC-ROC: 0.9710

# --- Average Metrics Across Folds ---
# Accuracy: 0.9141 ± 0.0022
# Precision: 0.8946 ± 0.0022
# Recall: 0.9450 ± 0.0016
# F1: 0.9191 ± 0.0018
# Auc_roc: 0.9720 ± 0.0010

# Training final model on combined training + validation data...

# Evaluating on test set...

# --- Test Set Metrics ---
# Accuracy: 0.9160
# Precision: 0.8908
# Recall: 0.9520
# F1: 0.9204
# Auc_roc: 0.9730

# Computing SHAP values for model interpretability...
# shap_values[1] shape: (8, 2)
# X_test shape: (8941, 8)
# Traceback (most recent call last):
#   File "D:\Karma-AI\Classifier.py", line 153, in <module>
#     shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False, random_state=rng)
# TypeError: summary_legacy() got an unexpected keyword argument 'random_state'





# import json
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import shap
# import joblib
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Set random seed for reproducibility (aligned with dataset generation)
# np.random.seed(42)

# # Load datasets
# def load_dataset(filename):
#     with open(filename, 'r') as f:
#         data = json.load(f)
#     X = pd.DataFrame([s["features"] for s in data])
#     y = np.array([s["label"] for s in data])
#     return X, y

# print("Loading datasets...")
# X_train, y_train = load_dataset("training_data.json")
# X_val, y_val = load_dataset("validation_data.json")
# X_test, y_test = load_dataset("testing_data.json")

# # Combine training and validation for cross-validation
# X_train_val = pd.concat([X_train, X_val], axis=0)
# y_train_val = np.concatenate([y_train, y_val])

# # Define K-Fold cross-validation
# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Initialize lists to store metrics for each fold
# fold_metrics = {
#     "accuracy": [],
#     "precision": [],
#     "recall": [],
#     "f1": [],
#     "auc_roc": []
# }

# # Initialize model
# model = RandomForestClassifier(
#     n_estimators=150,
#     max_depth=10,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     random_state=42,
#     n_jobs=-1
# )

# # Perform K-Fold cross-validation
# print(f"Starting {n_splits}-fold cross-validation...")
# fold_num = 1
# for train_idx, val_idx in kf.split(X_train_val):
#     print(f"\nFold {fold_num}/{n_splits}")
    
#     # Split data
#     X_fold_train, X_fold_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
#     y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
    
#     # Train model
#     model.fit(X_fold_train, y_fold_train)
    
#     # Predict on validation fold
#     y_pred = model.predict(X_fold_val)
#     y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    
#     # Calculate metrics
#     accuracy = accuracy_score(y_fold_val, y_pred)
#     precision = precision_score(y_fold_val, y_pred)
#     recall = recall_score(y_fold_val, y_pred)
#     f1 = f1_score(y_fold_val, y_pred)
#     auc_roc = roc_auc_score(y_fold_val, y_pred_proba)
    
#     # Store metrics
#     fold_metrics["accuracy"].append(accuracy)
#     fold_metrics["precision"].append(precision)
#     fold_metrics["recall"].append(recall)
#     fold_metrics["f1"].append(f1)
#     fold_metrics["auc_roc"].append(auc_roc)
    
#     # Print metrics for this fold
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1-Score: {f1:.4f}")
#     print(f"AUC-ROC: {auc_roc:.4f}")
    
#     fold_num += 1

# # Calculate and print average metrics across folds
# print("\n--- Average Metrics Across Folds ---")
# for metric, values in fold_metrics.items():
#     avg = np.mean(values)
#     std = np.std(values)
#     print(f"{metric.capitalize()}: {avg:.4f} ± {std:.4f}")

# # Train final model on combined training + validation data
# print("\nTraining final model on combined training + validation data...")
# model.fit(X_train_val, y_train_val)

# # Evaluate on test set
# print("\nEvaluating on test set...")
# y_test_pred = model.predict(X_test)
# y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# test_metrics = {
#     "accuracy": accuracy_score(y_test, y_test_pred),
#     "precision": precision_score(y_test, y_test_pred),
#     "recall": recall_score(y_test, y_test_pred),
#     "f1": f1_score(y_test, y_test_pred),
#     "auc_roc": roc_auc_score(y_test, y_test_pred_proba)
# }

# print("\n--- Test Set Metrics ---")
# for metric, value in test_metrics.items():
#     print(f"{metric.capitalize()}: {value:.4f}")

# # SHAP Explainer for feature importance
# print("\nComputing SHAP values for model interpretability...")
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)

# # Plot SHAP summary (feature importance)
# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
# plt.title("Feature Importance (SHAP Values) on Test Set")
# plt.tight_layout()
# plt.savefig("shap_feature_importance.png")
# plt.close()

# # Plot SHAP summary (detailed impact)
# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values[1], X_test, show=False)
# plt.title("SHAP Summary Plot on Test Set")
# plt.tight_layout()
# plt.savefig("shap_summary_plot.png")
# plt.close()

# # Save the model
# print("\nSaving the trained model...")
# joblib.dump(model, "classifier.pkl")
# print("Model saved as 'classifier.pkl'")

# # Save metrics to a file
# metrics_report = {
#     "cross_validation_metrics": fold_metrics,
#     "average_metrics": {metric: np.mean(values) for metric, values in fold_metrics.items()},
#     "test_metrics": test_metrics,
#     "timestamp": str(datetime.now())
# }

# with open("training_metrics.json", "w") as f:
#     json.dump(metrics_report, f, indent=2)
# print("Metrics saved to 'training_metrics.json'")
# import json
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_sample_weight
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter, defaultdict
# import logging
# from dataset import parse_condition, check_condition, compute_activity_score, conditions, MODEL_FEATURE_KEYS, CONFIG_KARMA_MIN, CONFIG_KARMA_MAX

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Set random seed for reproducibility
# np.random.seed(42)

# # Load data with error handling
# def load_data(filename):
#     try:
#         with open(filename, "r") as f:
#             data = json.load(f)
#         X = pd.DataFrame([sample["features"] for sample in data])[MODEL_FEATURE_KEYS]
#         y = np.array([sample["label"] for sample in data])
#         additional_info = [{
#             "user_id": sample.get("user_id", f"unknown_{i:04d}"),
#             "day": sample.get("day", "2024-01-01"),
#             "reward_score": sample["reward_score"],
#             "box_type": sample["box_type"],
#             "rarity": sample["rarity"],
#             "reason": sample.get("reason", "No reward")
#         } for i, sample in enumerate(data)]
#         logger.info(f"Loaded {filename}: {X.shape[0]} samples, {X.shape[1]} features")
#         return X, y, additional_info
#     except FileNotFoundError:
#         logger.error(f"{filename} not found.")
#         raise
#     except json.JSONDecodeError:
#         logger.error(f"Invalid JSON in {filename}.")
#         raise

# # Load datasets
# try:
#     X_train, y_train, train_info = load_data("training_data.json")
#     X_test, y_test, test_info = load_data("testing_data.json")
# except Exception as e:
#     logger.error(f"Failed to load data: {e}")
#     exit(1)

# # Print data shapes and label distribution
# logger.info(f"Training data shape: {X_train.shape}")
# logger.info(f"Testing data shape: {X_test.shape}")
# logger.info("\nTraining Label Distribution:")
# logger.info(pd.Series(y_train).value_counts().to_string())

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Handle class imbalance with sample weights
# weights = compute_sample_weight('balanced', y_train)
# logger.info("Computed sample weights for class imbalance")

# # Define hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Initialize RandomForestClassifier
# rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

# # Custom cross-validation
# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# fold_metrics = defaultdict(list)
# fold_confusion_matrices = []

# # Cross-validation loop
# for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
#     logger.info(f"\nTraining Fold {fold}/{n_splits}")
#     X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#     y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
#     fold_weights = weights[train_idx]

#     # Hyperparameter tuning
#     grid_search = GridSearchCV(
#         estimator=rf,
#         param_grid=param_grid,
#         cv=3,
#         scoring='f1_weighted',
#         n_jobs=-1,
#         verbose=0
#     )
#     grid_search.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)

#     # Best model for this fold
#     best_model = grid_search.best_estimator_
#     y_fold_pred = best_model.predict(X_fold_val)
#     y_fold_pred_proba = best_model.predict_proba(X_fold_val)[:, 1]

#     # Store metrics
#     fold_metrics['fold'].append(fold)
#     fold_metrics['accuracy'].append(accuracy_score(y_fold_val, y_fold_pred))
#     fold_metrics['f1_score'].append(classification_report(y_fold_val, y_fold_pred, output_dict=True)['weighted avg']['f1-score'])
#     fold_metrics['best_params'].append(grid_search.best_params_)
#     fold_confusion_matrices.append(confusion_matrix(y_fold_val, y_fold_pred))

#     # Log fold performance
#     logger.info(f"Fold {fold} Accuracy: {fold_metrics['accuracy'][-1]:.4f}")
#     logger.info(f"Fold {fold} F1-Score: {fold_metrics['f1_score'][-1]:.4f}")
#     logger.info(f"Fold {fold} Best Hyperparameters: {grid_search.best_params_}")
#     logger.info(f"Fold {fold} Classification Report:\n{classification_report(y_fold_val, y_fold_pred)}")

# # Summarize fold-wise performance
# fold_metrics_df = pd.DataFrame(fold_metrics)
# logger.info("\nCross-Validation Fold-wise Summary:")
# logger.info(fold_metrics_df.to_string())

# # Visualize fold-wise metrics
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.barplot(x='fold', y='accuracy', data=fold_metrics_df)
# plt.title('Fold-wise Accuracy')
# plt.subplot(1, 2, 2)
# sns.barplot(x='fold', y='f1_score', data=fold_metrics_df)
# plt.title('Fold-wise F1-Score')
# plt.tight_layout()
# plt.savefig("fold_metrics.png")
# plt.close()

# # Visualize fold-wise confusion matrices
# plt.figure(figsize=(15, 3))
# for i, cm in enumerate(fold_confusion_matrices, 1):
#     plt.subplot(1, n_splits, i)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Fold {i} Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
# plt.tight_layout()
# plt.savefig("fold_confusion_matrices.png")
# plt.close()

# # Train final model
# logger.info("\nTraining final model on full training data...")
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     scoring='f1_weighted',
#     n_jobs=-1,
#     verbose=1
# )
# grid_search.fit(X_train_scaled, y_train, sample_weight=weights)

# # Best model
# best_rf = grid_search.best_estimator_
# logger.info(f"\nBest Hyperparameters (Final Model): {grid_search.best_params_}")

# # Predict on test data
# y_pred = best_rf.predict(X_test_scaled)
# y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

# # Evaluate test performance
# logger.info("\nModel Performance on Testing Data:")
# logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# logger.info("\nClassification Report:")
# logger.info(classification_report(y_test, y_pred))

# # Confusion Matrix for Test Data
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Test Data Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig("test_confusion_matrix.png")
# plt.close()

# # ROC Curve and AUC
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig("roc_curve.png")
# plt.close()

# # Feature Importance
# feature_importances = best_rf.feature_importances_
# importance_df = pd.DataFrame({
#     'Feature': MODEL_FEATURE_KEYS,
#     'Importance': feature_importances
# }).sort_values('Importance', ascending=False)
# logger.info("\nFeature Importances:")
# logger.info(importance_df.to_string())

# # Visualize Feature Importance
# plt.figure(figsize=(8, 6))
# sns.barplot(x='Importance', y='Feature', data=importance_df)
# plt.title('Feature Importance')
# plt.savefig("feature_importance.png")
# plt.close()

# # Probability Threshold Tuning
# thresholds = [0.4, 0.5]
# logger.info("\nPerformance at Different Probability Thresholds:")
# for threshold in thresholds:
#     y_pred_adj = (y_pred_proba >= threshold).astype(int)
#     logger.info(f"\nThreshold = {threshold}:")
#     logger.info(classification_report(y_test, y_pred_adj))

# # Assign reward details
# def assign_reward_details(metrics, conditions):
#     metrics_dict = metrics.to_dict()
#     user_state = {"recent_rewards": 0, "last_reward_day": -1}
#     rarity_weights = {"common": 1.0, "rare": 0.8, "epic": 0.6, "legendary": 0.5}
#     reward_freq_factor = 1.0

#     shuffled_conditions = conditions.copy()
#     np.random.shuffle(shuffled_conditions)

#     for condition in shuffled_conditions:
#         if condition["parsed_condition"] is None:
#             continue
#         if check_condition(metrics_dict, condition["parsed_condition"]):
#             prob = float(condition.get("probability", 1.0)) * reward_freq_factor * rarity_weights.get(condition["rarity"], 1.0)
#             if np.random.random() < prob:
#                 if int(condition["label"]) == 1:
#                     base_score = int(condition["reward_score"])
#                     activity_score = compute_activity_score(metrics_dict)
#                     score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
#                     reward_score = np.random.randint(
#                         max(CONFIG_KARMA_MIN, base_score - 5),
#                         min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
#                     )
#                     return {
#                         "label": 1,
#                         "reward_score": reward_score,
#                         "box_type": condition["box_type"],
#                         "rarity": condition["rarity"],
#                         "reason": condition["reason"]
#                     }
#     is_somewhat_active = metrics_dict.get("login_streak", 0) > 0 and (
#         metrics_dict.get("karma_earned_today", 0) > 0 or
#         any(metrics_dict.get(k, 0) > 0 for k in ["posts_created", "comments_written", "quizzes_completed", "buddies_messaged"])
#     )
#     if is_somewhat_active and np.random.random() < 0.03:
#         return {
#             "label": 1,
#             "reward_score": np.random.randint(CONFIG_KARMA_MIN, CONFIG_KARMA_MIN + 2),
#             "box_type": "mystery",
#             "rarity": "common",
#             "reason": "Low-level activity"
#         }
#     return {
#         "label": 0,
#         "reward_score": 0,
#         "box_type": None,
#         "rarity": None,
#         "reason": "No reward"
#     }

# # Predict and assign rewards
# predictions = []
# for i, (idx, metrics) in enumerate(X_test.iterrows()):
#     pred_proba = best_rf.predict_proba(X_test_scaled[i:i+1])[:, 1][0]
#     pred_label = 1 if pred_proba >= 0.5 else 0  # Default threshold
#     reward_details = assign_reward_details(metrics, conditions)
#     predictions.append({
#         "user_id": test_info[i]["user_id"],
#         "day": test_info[i]["day"],
#         "features": metrics.to_dict(),
#         "true_label": int(y_test[i]),
#         "predicted_label": pred_label,
#         "predicted_proba": float(pred_proba),
#         "reward_score": reward_details["reward_score"],
#         "box_type": reward_details["box_type"],
#         "rarity": reward_details["rarity"],
#         "reason": reward_details["reason"]
#     })

# # Save predictions
# with open("predictions.json", "w") as f:
#     json.dump(predictions, f, indent=2)

# # Prediction stats
# pred_box_types = Counter(pred["box_type"] for pred in predictions if pred["predicted_label"] == 1)
# pred_rarities = Counter(pred["rarity"] for pred in predictions if pred["predicted_label"] == 1)
# logger.info("\n--- Prediction Stats ---")
# logger.info(f"Total Predictions: {len(predictions)}")
# logger.info(f"Predicted Rewarded Samples: {sum(1 for p in predictions if p['predicted_label'] == 1)} "
#             f"({sum(1 for p in predictions if p['predicted_label'] == 1)/len(predictions)*100:.2f}%)")
# logger.info(f"Predicted Box Type Distribution: {dict(pred_box_types)}")
# logger.info(f"Predicted Rarity Distribution: {dict(pred_rarities)}")

# # Example predictions
# logger.info("\nExample Predicted Rewarded Sample:")
# for pred in predictions:
#     if pred["predicted_label"] == 1:
#         logger.info(f"\n{json.dumps(pred, indent=2)}")
#         break
# logger.info("\nExample Predicted Non-Rewarded Sample:")
# for pred in predictions:
#     if pred["predicted_label"] == 0:
#         logger.info(f"\n{json.dumps(pred, indent=2)}")
#         break

# # Save model and feature names
# joblib.dump(best_rf, 'classifier.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# feature_names = MODEL_FEATURE_KEYS
# with open('feature_names.json', 'w') as f:
#     json.dump(feature_names, f)
# logger.info("\nSaved model as 'classifier.pkl', scaler as 'scaler.pkl', and feature names as 'feature_names.json'.")

# # Single-user prediction
# test_user_metrics = {
#     "login_streak": 5,
#     "posts_created": 2,
#     "comments_written": 3,
#     "upvotes_received": 25,
#     "quizzes_completed": 2,
#     "buddies_messaged": 1,
#     "karma_spent": 10,
#     "karma_earned_today": 20
# }
# test_df = pd.DataFrame([test_user_metrics], columns=MODEL_FEATURE_KEYS)
# test_scaled = scaler.transform(test_df)
# pred_proba = best_rf.predict_proba(test_scaled)[:, 1][0]
# pred_label = 1 if pred_proba >= 0.5 else 0
# reward_details = assign_reward_details(test_df.iloc[0], conditions)
# logger.info("\nTest Prediction for Single User:")
# logger.info({
#     "user_id": "test_user_5000",
#     "day": "2024-01-25",
#     "features": test_user_metrics,
#     "predicted_label": pred_label,
#     "predicted_proba": pred_proba,
#     **reward_details
# })

# # Feature Importance Chart (Chart.js)
# chart_feature_importance = {
#     "type": "bar",
#     "data": {
#         "labels": importance_df["Feature"].tolist(),
#         "datasets": [{
#             "label": "Feature Importance",
#             "data": importance_df["Importance"].tolist(),
#             "backgroundColor": "#4CAF50",
#             "borderColor": "#388E3C",
#             "borderWidth": 1
#         }]
#     },
#     "options": {
#         "scales": {
#             "y": {
#                 "beginAtZero": True,
#                 "title": {"display": True, "text": "Importance Score"}
#             },
#             "x": {
#                 "title": {"display": True, "text": "Feature"}
#             }
#         },
#         "plugins": {
#             "legend": {"display": False},
#             "title": {"display": True, "text": "Feature Importance for RandomForestClassifier"}
#         }
#     }
# }

# # Predicted Rarity Distribution Chart (Chart.js)
# chart_rarity_distribution = {
#     "type": "pie",
#     "data": {
#         "labels": list(pred_rarities.keys()),
#         "datasets": [{
#             "label": "Predicted Rarity Distribution",
#             "data": list(pred_rarities.values()),
#             "backgroundColor": ["#4CAF50", "#FF9800", "#F44336", "#9C27B0"],
#             "borderColor": ["#388E3C", "#F57C00", "#D32F2F", "#7B1FA2"],
#             "borderWidth": 1
#         }]
#     },
#     "options": {
#         "plugins": {
#             "legend": {"position": "right"},
#             "title": {"display": true, "text": "Predicted Rarity Distribution"}
#         }
#     }
# }

# # Save charts
# with open("feature_importance_chart.json", "w") as f:
#     json.dump(chart_feature_importance, f, indent=2)
# with open("rarity_distribution_chart.json", "w") as f:
#     json.dump(chart_rarity_distribution, f, indent=2)
# import json
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
# from sklearn.utils.class_weight import compute_sample_weight
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict

# # Set random seed for reproducibility
# np.random.seed(42)

# # Load training and testing datasets
# with open('training_data.json', 'r') as f:
#     training_data = json.load(f)

# with open('testing_data.json', 'r') as f:
#     testing_data = json.load(f)

# # Extract features and labels
# X_train = pd.DataFrame([data_point['features'] for data_point in training_data])
# y_train = np.array([data_point['label'] for data_point in training_data])
# X_test = pd.DataFrame([data_point['features'] for data_point in testing_data])
# y_test = np.array([data_point['label'] for data_point in testing_data])

# # Print data shapes and label distribution
# print(f"Training data shape: {X_train.shape}")
# print(f"Testing data shape: {X_test.shape}")
# print("\nTraining Label Distribution:")
# print(pd.Series(y_train).value_counts())

# # Handle class imbalance with sample weights
# weights = compute_sample_weight('balanced', y_train)

# # Define hyperparameter grid for tuning
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Initialize Random Forest Classifier
# rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

# # Custom cross-validation with detailed fold-wise metrics
# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# fold_metrics = defaultdict(list)
# fold_confusion_matrices = []

# # Manual cross-validation loop for transparency
# for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
#     print(f"\nTraining Fold {fold}/{n_splits}")
#     X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#     y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
#     fold_weights = weights[train_idx]
    
#     # Train model on this fold
#     grid_search = GridSearchCV(
#         estimator=rf,
#         param_grid=param_grid,
#         cv=3,  # Inner CV for hyperparameter tuning
#         scoring='f1_weighted',
#         n_jobs=-1,
#         verbose=0
#     )
#     grid_search.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)
    
#     # Best model for this fold
#     best_model = grid_search.best_estimator_
#     y_fold_pred = best_model.predict(X_fold_val)
#     y_fold_pred_proba = best_model.predict_proba(X_fold_val)[:, 1]
    
#     # Store metrics
#     fold_metrics['fold'].append(fold)
#     fold_metrics['accuracy'].append(accuracy_score(y_fold_val, y_fold_pred))
#     fold_metrics['f1_score'].append(classification_report(y_fold_val, y_fold_pred, output_dict=True)['weighted avg']['f1-score'])
#     fold_metrics['best_params'].append(grid_search.best_params_)
    
#     # Store confusion matrix
#     cm = confusion_matrix(y_fold_val, y_fold_pred)
#     fold_confusion_matrices.append(cm)
    
#     # Print fold performance
#     print(f"Fold {fold} Accuracy: {fold_metrics['accuracy'][-1]:.4f}")
#     print(f"Fold {fold} F1-Score: {fold_metrics['f1_score'][-1]:.4f}")
#     print(f"Fold {fold} Best Hyperparameters: {grid_search.best_params_}")
#     print(f"Fold {fold} Classification Report:")
#     print(classification_report(y_fold_val, y_fold_pred))

# # Summarize fold-wise performance
# fold_metrics_df = pd.DataFrame(fold_metrics)
# print("\nCross-Validation Fold-wise Summary:")
# print(fold_metrics_df)

# # Visualize fold-wise metrics
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.barplot(x='fold', y='accuracy', data=fold_metrics_df)
# plt.title('Fold-wise Accuracy')
# plt.subplot(1, 2, 2)
# sns.barplot(x='fold', y='f1_score', data=fold_metrics_df)
# plt.title('Fold-wise F1-Score')
# plt.tight_layout()
# plt.show()

# # Visualize fold-wise confusion matrices
# plt.figure(figsize=(15, 3))
# for i, cm in enumerate(fold_confusion_matrices, 1):
#     plt.subplot(1, n_splits, i)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Fold {i} Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
# plt.tight_layout()
# plt.show()

# # Train final model on full training data
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     scoring='f1_weighted',
#     n_jobs=-1,
#     verbose=1
# )
# grid_search.fit(X_train, y_train, sample_weight=weights)

# # Get the best model
# best_rf = grid_search.best_estimator_
# print("\nBest Hyperparameters (Final Model):", grid_search.best_params_)

# # Make predictions on the testing data
# y_pred = best_rf.predict(X_test)
# y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# # Evaluate performance with multiple metrics
# print("\nModel Performance on Testing Data:")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix for Test Data
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Test Data Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # ROC Curve and AUC for Test Data
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

# # Feature Importance Analysis
# feature_importances = best_rf.feature_importances_
# importance_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': feature_importances
# }).sort_values('Importance', ascending=False)
# print("\nFeature Importances:")
# print(importance_df)

# # Visualize Feature Importance
# plt.figure(figsize=(8, 6))
# sns.barplot(x='Importance', y='Feature', data=importance_df)
# plt.title('Feature Importance')
# plt.show()

# # Probability Threshold Tuning
# thresholds = [0.4, 0.5, 0.6, 0.7]
# print("\nPerformance at Different Probability Thresholds:")
# for threshold in thresholds:
#     y_pred_adj = (y_pred_proba >= threshold).astype(int)
#     print(f"\nThreshold = {threshold}:")
#     print(classification_report(y_test, y_pred_adj))

# # Save the best model and feature names
# joblib.dump(best_rf, 'classifier.pkl')
# feature_names = X_train.columns.tolist()
# with open('feature_names.json', 'w') as f:
#     json.dump(feature_names, f)

# print("\nBest model saved as 'classifier.pkl' and feature names saved as 'feature_names.json'.")

# import json
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
# from sklearn.utils.class_weight import compute_sample_weight
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load training and testing datasets
# with open('training_data.json', 'r') as f:
#     training_data = json.load(f)

# with open('testing_data.json', 'r') as f:
#     testing_data = json.load(f)

# # Extract features and labels
# X_train = pd.DataFrame([data_point['features'] for data_point in training_data])
# y_train = np.array([data_point['label'] for data_point in training_data])
# X_test = pd.DataFrame([data_point['features'] for data_point in testing_data])
# y_test = np.array([data_point['label'] for data_point in testing_data])

# # Print data shapes and label distribution
# print(f"Training data shape: {X_train.shape}")
# print(f"Testing data shape: {X_test.shape}")
# print("\nTraining Label Distribution:")
# print(pd.Series(y_train).value_counts())

# # Handle class imbalance with sample weights
# weights = compute_sample_weight('balanced', y_train)

# # Define hyperparameter grid for tuning
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Initialize Random Forest Classifier
# rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     scoring='f1_weighted',  # Optimize for balanced performance
#     n_jobs=-1,
#     verbose=1
# )

# # Train the model with sample weights
# grid_search.fit(X_train, y_train, sample_weight=weights)

# # Get the best model
# best_rf = grid_search.best_estimator_
# print("\nBest Hyperparameters:", grid_search.best_params_)

# # Make predictions on the testing data
# y_pred = best_rf.predict(X_test)
# y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# # Evaluate performance with multiple metrics
# print("\nModel Performance on Testing Data:")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # ROC Curve and AUC
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

# # Feature Importance Analysis
# feature_importances = best_rf.feature_importances_
# importance_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': feature_importances
# }).sort_values('Importance', ascending=False)
# print("\nFeature Importances:")
# print(importance_df)

# # Visualize Feature Importance
# plt.figure(figsize=(8, 6))
# sns.barplot(x='Importance', y='Feature', data=importance_df)
# plt.title('Feature Importance')
# plt.show()

# # Probability Threshold Tuning
# thresholds = [0.4, 0.5, 0.6, 0.7]
# print("\nPerformance at Different Probability Thresholds:")
# for threshold in thresholds:
#     y_pred_adj = (y_pred_proba >= threshold).astype(int)
#     print(f"\nThreshold = {threshold}:")
#     print(classification_report(y_test, y_pred_adj))

# # Save the best model and feature names
# joblib.dump(best_rf, 'classifier.pkl')
# feature_names = X_train.columns.tolist()
# with open('feature_names.json', 'w') as f:
#     json.dump(feature_names, f)

# print("\nBest model saved as 'classifier.pkl' and feature names saved as 'feature_names.json'.")
# Training data shape: (50000, 8)
# Testing data shape: (10000, 8)

# Training Label Distribution:
# 1    38887
# 0    11113
# Name: count, dtype: int64
# Fitting 5 folds for each of 108 candidates, totalling 540 fits

# Best Hyperparameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}

# Model Performance on Testing Data:
# Accuracy: 0.9767

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.90      1.00      0.95      2165
#            1       1.00      0.97      0.98      7835

#     accuracy                           0.98     10000
#    macro avg       0.95      0.99      0.97     10000
# weighted avg       0.98      0.98      0.98     10000


# Feature Importances:
#               Feature  Importance
# 6         karma_spent    0.218858
# 7  karma_earned_today    0.214305
# 0        login_streak    0.155422
# 4   quizzes_completed    0.153427
# 3    upvotes_received    0.078375
# 2    comments_written    0.076575
# 5    buddies_messaged    0.063500
# 1       posts_created    0.039537

# Performance at Different Probability Thresholds:

# Threshold = 0.4:
#               precision    recall  f1-score   support

#            0       0.97      0.98      0.97      2165
#            1       0.99      0.99      0.99      7835

#     accuracy                           0.99     10000
#    macro avg       0.98      0.98      0.98     10000
# weighted avg       0.99      0.99      0.99     10000


# Threshold = 0.5:
#               precision    recall  f1-score   support

#            0       0.90      1.00      0.95      2165
#            1       1.00      0.97      0.98      7835

#     accuracy                           0.98     10000
#    macro avg       0.95      0.99      0.97     10000
# weighted avg       0.98      0.98      0.98     10000


# Threshold = 0.6:
#               precision    recall  f1-score   support

#            0       0.79      1.00      0.88      2165
#            1       1.00      0.93      0.96      7835

#     accuracy                           0.94     10000
#    macro avg       0.89      0.96      0.92     10000
# weighted avg       0.95      0.94      0.94     10000


# Threshold = 0.7:
#               precision    recall  f1-score   support

#            0       0.62      1.00      0.76      2165
#            1       1.00      0.83      0.91      7835

#     accuracy                           0.86     10000
#    macro avg       0.81      0.91      0.83     10000
# weighted avg       0.92      0.86      0.87     10000











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