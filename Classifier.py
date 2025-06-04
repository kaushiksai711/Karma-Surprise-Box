import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from imblearn.over_sampling import SMOTE
import re
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
import seaborn as sns
# Define expected features (aligned with dataset.py)
MODEL_FEATURE_KEYS = ["login_streak", "posts_created", "comments_written", "upvotes_received", 
                      "quizzes_completed", "buddies_messaged", "karma_spent", "karma_earned_today"]

# Set random seed for reproducibility
np.random.seed(42)

# --- Functions from dataset.py for condition parsing and temporal trends ---
def tokenize_condition(condition_str):
    pattern = r'(\w+|>=|<=|==|<|>|\(|\)|and|or|\d+)'
    tokens = re.findall(pattern, condition_str)
    return [token.strip() for token in tokens if token.strip()]

def parse_atomic_condition(tokens, start_idx):
    if start_idx + 2 >= len(tokens):
        return None, start_idx
    var_token = tokens[start_idx]
    op_token = tokens[start_idx + 1]
    val_token = tokens[start_idx + 2]
    if (var_token in MODEL_FEATURE_KEYS and 
        op_token in ['>=', '<=', '==', '<', '>'] and 
        val_token.isdigit()):
        return (var_token, op_token, int(val_token)), start_idx + 3
    return None, start_idx

def parse_condition_recursive(tokens, start_idx=0):
    left_expr, idx = parse_term(tokens, start_idx)
    while idx < len(tokens):
        if tokens[idx] == 'or':
            right_expr, idx = parse_condition_recursive(tokens, idx + 1)
            left_expr = ('or', left_expr, right_expr)
        else:
            break
    return left_expr, idx

def parse_term(tokens, start_idx):
    left_expr, idx = parse_factor(tokens, start_idx)
    while idx < len(tokens) and tokens[idx] == 'and':
        right_expr, idx = parse_factor(tokens, idx + 1)
        left_expr = ('and', left_expr, right_expr)
    return left_expr, idx

def parse_factor(tokens, start_idx):
    if start_idx >= len(tokens):
        raise ValueError("Unexpected end of expression")
    if tokens[start_idx] == '(':
        expr, idx = parse_condition_recursive(tokens, start_idx + 1)
        if idx >= len(tokens) or tokens[idx] != ')':
            raise ValueError("Missing closing parenthesis")
        return expr, idx + 1
    else:
        atomic, idx = parse_atomic_condition(tokens, start_idx)
        if atomic is None:
            raise ValueError(f"Invalid atomic condition at position {start_idx}: {tokens[start_idx:start_idx+3]}")
        return atomic, idx

def parse_condition(condition_str):
    if not condition_str.strip():
        return None
    tokens = tokenize_condition(condition_str)
    if not tokens:
        return None
    try:
        expr, _ = parse_condition_recursive(tokens)
        return expr
    except Exception as e:
        print(f"Error parsing condition '{condition_str}': {e}")
        return None

def evaluate_expression(expr, metrics):
    if isinstance(expr, tuple) and len(expr) == 3:
        if expr[0] in MODEL_FEATURE_KEYS:
            variable, operator, value = expr
            metric_value = metrics.get(variable, 0)
            if operator == '>=':
                return metric_value >= value
            elif operator == '<=':
                return metric_value <= value
            elif operator == '==':
                return metric_value == value
            elif operator == '<':
                return metric_value < value
            elif operator == '>':
                return metric_value > value
            else:
                raise ValueError(f"Unknown operator: {operator}")
        elif expr[0] == 'and':
            _, left, right = expr
            return evaluate_expression(left, metrics) and evaluate_expression(right, metrics)
        elif expr[0] == 'or':
            _, left, right = expr
            return evaluate_expression(left, metrics) or evaluate_expression(right, metrics)
    raise ValueError(f"Invalid expression: {expr}")

def check_condition(metrics, parsed_condition):
    if parsed_condition is None:
        return False
    try:
        return evaluate_expression(parsed_condition, metrics)
    except Exception as e:
        print(f"Error evaluating condition: {e}")
        return False

def get_temporal_multiplier(day_of_week, month):
    with open("config.json", "r") as f:
        config = json.load(f)
    TEMPORAL_TRENDS = config["temporal_trends"]
    base_mult = 1.2 if day_of_week in [5, 6] else 1.0
    seasonal_mult = TEMPORAL_TRENDS.get("seasonal_multipliers", {}).get(str(month), 1.0)
    return base_mult * seasonal_mult

# Load conditions from conditions.csv
def load_conditions():
    conditions = []
    with open("conditions.csv", "r") as csvfile:
        reader = pd.read_csv(csvfile)
        for _, row in reader.iterrows():
            parsed_condition = parse_condition(row["condition"])
            if parsed_condition is None:
                print(f"Warning: Could not parse condition: '{row['condition']}'")
            row_dict = row.to_dict()
            row_dict["parsed_condition"] = parsed_condition
            conditions.append(row_dict)
    return [c for c in conditions if c["parsed_condition"] is not None]

# Load dataset with additional day and user_id for temporal and cluster features
def load_dataset(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    X = pd.DataFrame([s["features"] for s in data])
    y = np.array([s["label"] for s in data])
    days = [s["day"] for s in data]
    user_ids = [s["user_id"] for s in data]
    
    # Ensure only expected features are included and in the correct order
    missing_features = [feat for feat in MODEL_FEATURE_KEYS if feat not in X.columns]
    if missing_features:
        raise ValueError(f"Missing features in {filename}: {missing_features}")
    extra_features = [feat for feat in X.columns if feat not in MODEL_FEATURE_KEYS]
    if extra_features:
        print(f"Warning: Extra features in {filename}: {extra_features}. Dropping them.")
        X = X.drop(columns=extra_features)
    X = X[MODEL_FEATURE_KEYS]  # Reorder columns to match MODEL_FEATURE_KEYS
    return X, y, days, user_ids

# Add rule-based features
def add_rule_features(X, conditions):
    rule_features = np.zeros((len(X), len(conditions)))
    for i in range(len(X)):
        metrics = X.iloc[i].to_dict()
        for j, cond in enumerate(conditions):
            rule_features[i, j] = check_condition(metrics, cond["parsed_condition"])
    rule_cols = [f"rule_{i}" for i in range(len(conditions))]
    return pd.concat([X, pd.DataFrame(rule_features, columns=rule_cols, index=X.index)], axis=1)

# Add temporal features
def add_temporal_features(X, days):
    temporal_mults = []
    for day in days:
        day_dt = datetime.strptime(day, "%Y-%m-%d")
        mult = get_temporal_multiplier(day_dt.weekday(), day_dt.month)
        temporal_mults.append(mult)
    return pd.concat([X, pd.DataFrame({"temporal_multiplier": temporal_mults}, index=X.index)], axis=1)

# Assign clusters (same as HR-RFE)
def assign_clusters(user_ids, num_users=1000):
    with open("config.json", "r") as f:
        config = json.load(f)
    CLUSTER_PROBS = config["cluster probs"]
    user_clusters = np.random.choice(list(config["user_clusters"].keys()), size=num_users, p=CLUSTER_PROBS)
    user_to_cluster = {f"sim_user_{i:04d}": cluster for i, cluster in enumerate(user_clusters)}
    clusters = np.array([user_to_cluster[user_id] for user_id in user_ids])
    return clusters

# Apply SMOTE-R (adapted from HR-RFE)
def apply_smote_r(X, y, conditions):
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)
    label_1_conditions = [c for c in conditions if c["label"] == 1]
    
    # Adjust synthetic samples to satisfy at least one reward condition
    for i in range(len(X), len(X_res)):
        if y_res[i] == 1:
            metrics = {key: X_res.iloc[i][key] for key in MODEL_FEATURE_KEYS}
            while not any(check_condition(metrics, c["parsed_condition"]) for c in label_1_conditions):
                cond = np.random.choice(label_1_conditions)
                parsed_cond = cond["parsed_condition"]
                
                def adjust_metrics(expr, metrics):
                    if isinstance(expr, tuple) and len(expr) == 3:
                        if expr[0] in MODEL_FEATURE_KEYS:
                            var, op, val = expr
                            if op == ">=" and metrics[var] < val:
                                metrics[var] = val
                            elif op == "<=" and metrics[var] > val:
                                metrics[var] = val
                            elif op == "==" and metrics[var] != val:
                                metrics[var] = val
                            elif op == "<" and metrics[var] >= val:
                                metrics[var] = val - 1
                            elif op == ">" and metrics[var] <= val:
                                metrics[var] = val + 1
                        elif expr[0] == "and":
                            adjust_metrics(expr[1], metrics)
                            adjust_metrics(expr[2], metrics)
                        elif expr[0] == "or":
                            adjust_metrics(expr[1], metrics)
                    return metrics
                
                metrics = adjust_metrics(parsed_cond, metrics)
            
            for key in MODEL_FEATURE_KEYS:
                X_res.iloc[i, X_res.columns.get_loc(key)] = metrics[key]
            for j, cond in enumerate(conditions):
                X_res.iloc[i, X_res.columns.get_loc(f"rule_{j}")] = check_condition(metrics, cond["parsed_condition"])
    
    return X_res, y_res

# Find optimal threshold for accuracy
def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_accuracy = 0
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = thresh
    return best_threshold


def compute_per_category_metrics(y_true, y_pred, test_user_ids, test_days, conditions):
    # Load test data to get box_type and rarity
    with open("testing_data.json", "r") as f:
        test_data = json.load(f)
    
    # Map samples to their box_type and rarity
    sample_to_category = {}
    for sample in test_data:
        user_id = sample["user_id"]
        day = sample["day"]
        box_type = sample["box_type"]
        rarity = sample["rarity"]
        sample_key = (user_id, day)
        sample_to_category[sample_key] = {"box_type": box_type, "rarity": rarity}
    
    # Define reward type groups based on conditions.csv
    reward_type_map = {
        "streak": ["streak"],
        "engagement": ["engagement", "community_hero", "knowledge_expert", "comeback", "newcomer", "spotlight", "scholar"],
        "social": ["social", "SocialConnector", "community", "exchange", "supporter", "glue"]
    }
    
    # Initialize metrics dictionaries
    metrics_by_type = {rt: {"precision": [], "recall": [], "f1": []} for rt in reward_type_map}
    metrics_by_rarity = {r: {"precision": [], "recall": [], "f1": []} for r in ["common", "rare", "legendary"]}
    
    # Compute metrics for each category
    for rt, box_types in reward_type_map.items():
        y_true_rt = []
        y_pred_rt = []
        for i, (uid, day) in enumerate(zip(test_user_ids, test_days)):
            sample_key = (uid, day)
            if sample_key in sample_to_category and sample_to_category[sample_key]["box_type"] in box_types:
                y_true_rt.append(y_true[i])
                y_pred_rt.append(y_pred[i])
        if y_true_rt:
            metrics_by_type[rt]["precision"] = precision_score(y_true_rt, y_pred_rt, zero_division=0)
            metrics_by_type[rt]["recall"] = recall_score(y_true_rt, y_pred_rt, zero_division=0)
            metrics_by_type[rt]["f1"] = f1_score(y_true_rt, y_pred_rt, zero_division=0)
    
    for rarity in ["common", "rare", "legendary"]:
        y_true_r = []
        y_pred_r = []
        for i, (uid, day) in enumerate(zip(test_user_ids, test_days)):
            sample_key = (uid, day)
            if sample_key in sample_to_category and sample_to_category[sample_key]["rarity"] == rarity:
                y_true_r.append(y_true[i])
                y_pred_r.append(y_pred[i])
        if y_true_r:
            metrics_by_rarity[rarity]["precision"] = precision_score(y_true_r, y_pred_r, zero_division=0)
            metrics_by_rarity[rarity]["recall"] = recall_score(y_true_r, y_pred_r, zero_division=0)
            metrics_by_rarity[rarity]["f1"] = f1_score(y_true_r, y_pred_r, zero_division=0)
    
    return metrics_by_type, metrics_by_rarity

# Main execution
if __name__ == "__main__":
    print("Loading datasets...")
    X_train, y_train, train_days, train_user_ids = load_dataset("training_data.json")
    X_val, y_val, val_days, val_user_ids = load_dataset("validation_data.json")
    X_test, y_test, test_days, test_user_ids = load_dataset("testing_data.json")

# Debug: Check number of samples in each dataset
    print(f"X_train samples: {len(X_train)}")
    print(f"X_val samples: {len(X_val)}")
    print(f"X_test samples: {len(X_test)}")

# Load conditions
    conditions = load_conditions()
    print(f"Loaded {len(conditions)} valid conditions.")

# Add rule-based features
    X_train = add_rule_features(X_train, conditions)
    X_val = add_rule_features(X_val, conditions)
    X_test = add_rule_features(X_test, conditions)

# Add temporal features
    X_train = add_temporal_features(X_train, train_days)
    X_val = add_temporal_features(X_val, val_days)
    X_test = add_temporal_features(X_test, test_days)

# Assign clusters
    train_clusters = assign_clusters(train_user_ids)
    val_clusters = assign_clusters(val_user_ids)
    test_clusters = assign_clusters(test_user_ids)

# Combine training and validation for cross-validation
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val])
    train_val_clusters = np.concatenate([train_clusters, val_clusters])

# Debug: Print shapes and columns
    print(f"X_train_val shape: {X_train_val.shape}")
    print(f"X_train_val columns: {list(X_train_val.columns)}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_test columns: {list(X_test.columns)}")

# Define K-Fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics for each fold
    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_roc": []
    }

# Hyperparameter grid for Random Forest
    param_grid = {
        "n_estimators": [200],
        "max_depth": [50],
        "min_samples_split": [5],
        "min_samples_leaf": [5]
    }

# Perform K-Fold cross-validation with grid search
    print(f"Starting {n_splits}-fold cross-validation with grid search...")
    fold_num = 1
    best_models = []
    for train_idx, val_idx in kf.split(X_train_val):
        print(f"\nFold {fold_num}/{n_splits}")
    
    # Split data
        X_fold_train, X_fold_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
        clusters_fold_train = train_val_clusters[train_idx]
    
    # Apply SMOTE-R
    X_fold_train, y_fold_train = apply_smote_r(X_fold_train, y_fold_train, conditions)
    
    # Grid search for this fold
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_fold_train, y_fold_train)
    
    model = grid_search.best_estimator_
    best_models.append(model)
    
    # Find optimal threshold
    y_fold_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    threshold = find_optimal_threshold(y_fold_val, y_fold_pred_proba)
    
    # Predict on validation fold with optimal threshold
    y_pred = (y_fold_pred_proba >= threshold).astype(int)
    y_pred_proba = y_fold_pred_proba
    
    # Calculate metrics
    accuracy = accuracy_score(y_fold_val, y_pred)
    precision = precision_score(y_fold_val, y_pred)
    recall = recall_score(y_fold_val, y_pred)
    f1 = f1_score(y_fold_val, y_pred)
    auc_roc = roc_auc_score(y_fold_val, y_pred_proba)
    
    # Store metrics
    fold_metrics["accuracy"].append(accuracy)
    fold_metrics["precision"].append(precision)
    fold_metrics["recall"].append(recall)
    fold_metrics["f1"].append(f1)
    fold_metrics["auc_roc"].append(auc_roc)
    
    # Print metrics for this fold
    print(f"Best params: {grid_search.best_params_}")
    print(f"Optimal threshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    fold_num += 1

# Calculate and print average metrics across folds
    print("\n--- Average Metrics Across Folds ---")
    for metric, values in fold_metrics.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"{metric.capitalize()}: {avg:.4f} ± {std:.4f}")

# Train final model on combined training + validation data
    print("\nTraining final model on combined training + validation data...")
    X_train_val, y_train_val = apply_smote_r(X_train_val, y_train_val, conditions)
    final_model = RandomForestClassifier(
        n_estimators=300, max_depth=100, min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    final_model.fit(X_train_val, y_train_val)

# Find optimal threshold on validation data
    y_val_pred_proba = final_model.predict_proba(X_val)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val, y_val_pred_proba)
    print(f"Final optimal threshold: {optimal_threshold:.2f}")
    joblib.dump(final_model,"classifier.pkl")

# # Compute confusion matrix
# cm = confusion_matrix(y_test, y_test_pred)

# # Compute per-reward-type and rarity metrics
# metrics_by_reward_type, metrics_by_rarity = compute_per_category_metrics(
#     y_test_pred, y_test_pred, test_user_ids, test_days, conditions
# )

# # Plot calibration curve
# plt.figure(figsize=(8, 6))
# CalibrationDisplay.from_predictions(y_test, y_test_pred_proba, n_bins=10, ax=plt.gca())
# plt.title("Calibration Curve (Test Set)")
# plt.savefig("calibration_curve.png")
# plt.close()

# # Check if calibration is needed (reliability curve deviation)
# if not (0.95 <= np.mean(np.abs(precision - recall)) <= 1.05):  # Arbitrary threshold for miscalibration
#     print("Model is miscalibrated. Applying Platt scaling...")
#     calibrator = CalibratedClassifierCV(
#         final_model, method="sigmoid", cv="prefit"
#     )
#     calibrator.fit(X_val, y_val)  # Use validation data for calibration
    
#     # Re-evaluate with calibrated model
#     y_test_pred_proba_cal = calibrator.predict_proba(X_test)[:, 0]
#     y_test_pred_cal = (y_test_pred_proba_cal >= optimal_threshold).astype(int)
    
#     test_metrics_cal = {
#         "accuracy": accuracy_score(y_test, y_test_pred_cal),
#         "precision": precision_score(y_test, y_test_pred_cal),
#         "recall": recall_score(y_test, y_test_pred_cal),
#         "f1": f1_score(y_test, y_test_pred_cal),
#         "auc_roc": roc_auc_score(y_test, y_test_pred_proba_cal),
#         "auc_pr": auc(*precision_recall_curve(y_test, y_test_pred_proba_cal)[1::-1])
#     }
    
#     # Plot calibrated curve
#     plt.figure(figsize=(8, 6))
#     CalibrationDisplay.from_predictions(y_test, y_test_pred_proba_cal, n_bins=10, ax=plt.gca())
#     plt.title("Calibrated Curve (Test Set)")
#     plt.savefig("calibrated_curve.png")
#     plt.close()
    
#     # Save calibrated model
#     joblib.dump(calibrator, "calibrated_classifier.pkl")
#     print("Calibrated model saved as 'calibrated_classifier.pkl'")
# else:
#     test_metrics_cal = None
#     print("Model is well-calibrated. No calibration applied.")

# # Plot confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix (Test Set)")
# plt.ylabel("True Label")
# plt.xlabel("Predicted Label")
# plt.savefig("confusion_matrix.png")
# plt.close()

# # Print metrics
# print("\n--- Test Set Metrics ---")
# for metric, value in test_metrics.items():
#     print(f"{metric.capitalize()}: {value:.4f}")

# if test_metrics_cal:
#     print("\n--- Calibrated Test Set Metrics ---")
#     for metric, value in test_metrics_cal.items():
#         print(f"{metric.capitalize()}: {value:.4f}")

# print("\n--- Metrics by Reward Type ---")
# for rt, metrics in metrics_by_reward_type.items():
#     print(f"{rt.capitalize()}:")
#     for metric, value in metrics.items():
#         print(f"  {metric.capitalize()}: {value:.4f}")

# print("\n--- Metrics by Rarity ---")
# for rarity, metrics in metrics_by_rarity.items():
#     print(f"{rarity.capitalize()}:")
#     for metric, value in metrics.items():
#         print(f"  {metric.capitalize()}: {value:.4f}")

# # Update metrics report
# metrics_report = {
#     "cross_validation_metrics": fold_metrics,
#     "average_metrics": {metric: np.mean(values) for metric, values in fold_metrics.items()},
#     "test_metrics": test_metrics,
#     "test_metrics_calibrated": test_metrics_cal,
#     "metrics_by_reward_type": metrics_by_reward_type,
#     "metrics_by_rarity": metrics_by_rarity,
#     "confusion_matrix": cm.tolist(),
#     "timestamp": str(datetime.now()),
# }

# with open("training_metrics.json", "w") as f:
#     json.dump(metrics_report, f, indent=2);

# print("Metrics saved to 'training_metrics.json'")