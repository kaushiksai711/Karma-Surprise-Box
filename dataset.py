import csv
import re
import random
import json
import datetime
from collections import Counter
import numpy as np
import pandas as pd
from functools import lru_cache

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load config.json
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    CONFIG_KARMA_MIN = config["karma_min"]
    CONFIG_KARMA_MAX = config["karma_max"]
    METRIC_RANGES = config["metric_ranges"]
    USER_CLUSTERS = config["user_clusters"]
    CLUSTER_PROBS = config["cluster_probs"]
    TARGET_RARITY_DIST = config["target_rarity_dist"]
    TEMPORAL_TRENDS = config["temporal_trends"]
except FileNotFoundError:
    print("Error: config.json not found. Exiting.")
    exit(1)

MODEL_FEATURE_KEYS = ["login_streak", "posts_created", "comments_written", "upvotes_received", "quizzes_completed", "buddies_messaged", "karma_spent", "karma_earned_today"]

# Temporal trend multipliers
def get_temporal_multiplier(day_of_week, month):
    base_mult = 1.2 if day_of_week in [5, 6] else 1.0
    seasonal_mult = TEMPORAL_TRENDS.get("seasonal_multipliers", {}).get(str(month), 1.0)
    return base_mult * seasonal_mult

# Parse condition strings
@lru_cache(maxsize=1000)
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

def generate_daily_activity_metrics(cluster, day_of_week, month, previous_day_metrics=None, user_state=None):
    metrics = {}
    cluster_params = USER_CLUSTERS[cluster]
    temporal_mult = get_temporal_multiplier(day_of_week, month)
    
    for metric_name, (min_val, max_val) in METRIC_RANGES.items():
        prob_active, mean_val = cluster_params.get(metric_name, (0.5, 5))
        mean_val = mean_val * temporal_mult
        if metric_name == "login_streak":
            if previous_day_metrics and previous_day_metrics.get("login_streak", 0) > 0:
                prob_continue = prob_active * (1.2 if user_state.get("recent_rewards", 0) > 0 else 1.0)
                if random.random() < prob_continue:
                    metrics[metric_name] = min(previous_day_metrics["login_streak"] + 1, max_val)
                else:
                    metrics[metric_name] = 0
            elif random.random() < prob_active:
                metrics[metric_name] = 1
            else:
                metrics[metric_name] = 0
        elif metric_name == "karma_earned_today":
            if random.random() < prob_active:
                base_karma = (
                    metrics.get("posts_created", 0) * 8 +
                    metrics.get("comments_written", 0) * 4 +
                    metrics.get("upvotes_received", 0) * 1.5 +
                    metrics.get("quizzes_completed", 0) * 12
                )
                value = min(int(base_karma + np.random.exponential(mean_val * 0.15)), max_val)
                value = int(value * random.uniform(0.95, 1.05))
                if random.random() < 0.002:
                    value = int(max_val * random.uniform(0.7, 0.8))
                metrics[metric_name] = max(min_val, min(value, max_val))
            else:
                metrics[metric_name] = 0
        else:
            if random.random() < prob_active:
                value = min(int(np.random.exponential(mean_val)), max_val)
                value = int(value * random.uniform(0.95, 1.05))
                if random.random() < 0.002:
                    value = int(max_val * random.uniform(0.7, 0.8))
                metrics[metric_name] = max(min_val, min(value, max_val))
            else:
                metrics[metric_name] = 0
        # Dependencies
        if metric_name == "upvotes_received" and metrics.get("posts_created", 0) == 0:
            metrics[metric_name] = 0
    # Ensure karma consistency
    if metrics["karma_earned_today"] > 15 and metrics["posts_created"] == 0 and metrics["comments_written"] == 0 and metrics["quizzes_completed"] == 0:
        metrics["karma_earned_today"] = 0
    return metrics

def compute_activity_score(metrics):
    weights = {key: 1.0 / len(METRIC_RANGES) for key in METRIC_RANGES}
    score = 0
    for metric, weight in weights.items():
        max_val = METRIC_RANGES[metric][1]
        score += weight * (metrics.get(metric, 0) / max_val if max_val > 0 else 0)
    return score

def determine_reward_from_metrics(daily_metrics, conditions, user_state):
    label = 0
    reward_score = 0
    box_type = None
    rarity = None
    reason = "No reward"
    display_reason = None

    activity_score = compute_activity_score(daily_metrics)
    rarity_weights = {"common": 1.0, "rare": 1.0, "elite": 1.0, "legendary": 1.0}

    shuffled_conditions = conditions.copy()
    random.shuffle(shuffled_conditions)

    for condition in shuffled_conditions:
        if condition["parsed_condition"] is None:
            continue
        if check_condition(daily_metrics, condition["parsed_condition"]):
            prob = float(condition.get("probability", 1.0)) * rarity_weights.get(condition["rarity"], 1.0)
            if random.random() < prob:
                label = int(condition["label"])
                if label == 1:
                    base_score = int(condition["reward_score"])
                    score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
                    reward_score = random.randint(
                        max(CONFIG_KARMA_MIN, base_score - 3),
                        min(CONFIG_KARMA_MAX, base_score + 3 + score_range)
                    )
                    box_type = condition["box_type"]
                    rarity = condition["rarity"]
                    reason = condition["reason"]
                    display_reason = condition["display_reason"]
                return label, reward_score, box_type, rarity, reason, display_reason

    # Default mystery reward for low activity
    is_somewhat_active = daily_metrics.get("login_streak", 0) > 0 and (
        daily_metrics.get("karma_earned_today", 0) > 0 or
        any(daily_metrics.get(k, 0) > 0 for k in MODEL_FEATURE_KEYS if k != "login_streak")
    )
    if is_somewhat_active and random.random() < 0.25:
        label = 1
        reward_score = random.randint(CONFIG_KARMA_MIN, CONFIG_KARMA_MIN + 5)
        box_type = "mystery"
        rarity = "common"
        reason = "Low-level activity"
        display_reason = "Thanks for being active today!"

    return label, reward_score, box_type, rarity, reason, display_reason

def validate_dataset(samples, dataset_name):
    issues = []
    df = pd.DataFrame([s["features"] for s in samples])
    for i, metrics in enumerate([s["features"] for s in samples]):
        if metrics["upvotes_received"] > 0 and metrics["posts_created"] == 0:
            issues.append(f"{dataset_name} Sample {i}: Upvotes without posts")
        if metrics["karma_earned_today"] > 15 and metrics["posts_created"] == 0 and metrics["comments_written"] == 0 and metrics["quizzes_completed"] == 0:
            issues.append(f"{dataset_name} Sample {i}: High karma without posts/comments/quizzes")
    for column in df.columns:
        skewness = df[column].skew()
        if abs(skewness) < 0.5 and column not in ["login_streak"]:
            issues.append(f"{dataset_name} Metric {column} too uniform (skewness: {skewness:.2f})")
    return issues

def generate_dataset(num_users, start_user_id, days_per_user, output_filename, start_date, conditions, user_clusters):
    all_samples = []
    reward_counts = {rarity: 0 for rarity in TARGET_RARITY_DIST}
    box_type_counts = Counter()

    print(f"Generating {output_filename} for {num_users} users, {days_per_user} days each...")

    for user_num in range(num_users):
        user_id = f"sim_user_{start_user_id + user_num:04d}"
        cluster = user_clusters[user_num]
        previous_day_data = None
        user_state = {"recent_rewards": 0, "last_reward_day": -1}
        if user_num % 10 == 0 and user_num > 0:
            print(f"Processing user {user_num}/{num_users} (Cluster: {cluster})...")

        for day_num in range(days_per_user):
            current_date = start_date + datetime.timedelta(days=day_num)
            day_of_week = current_date.weekday()
            month = current_date.month
            current_daily_metrics = generate_daily_activity_metrics(cluster, day_of_week, month, previous_day_data, user_state)
            label, reward_score, box_type, rarity, reason, display_reason = determine_reward_from_metrics(current_daily_metrics, conditions, user_state)
            features_for_model = {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}

            sample = {
                "user_id": user_id,
                "day": str(current_date),
                "features": features_for_model,
                "label": label,
                "reward_score": reward_score,
                "box_type": box_type,
                "rarity": rarity,
                "reason": reason,
                "display_reason": display_reason,
                "status": "delivered" if label == 1 else "missed"
            }
            all_samples.append(sample)

            if label == 1:
                user_state["recent_rewards"] += 1
                user_state["last_reward_day"] = day_num
                reward_counts[rarity] = reward_counts.get(rarity, 0) + 1
                box_type_counts[box_type] += 1
            if day_num - user_state["last_reward_day"] > 5:
                user_state["recent_rewards"] = max(0, user_state["recent_rewards"] - 1)
            previous_day_data = current_daily_metrics

    # Balance rarity distribution
    total_rewards = sum(reward_counts.values())
    target_samples = num_users * days_per_user
    if total_rewards > 0:
        for rarity, target_frac in TARGET_RARITY_DIST.items():
            current_frac = reward_counts.get(rarity, 0) / total_rewards
            if current_frac < target_frac * 0.95:
                print(f"Balancing {rarity} rewards in {output_filename} (current: {current_frac:.3f}, target: {target_frac:.3f})")
                extra_samples = []
                max_extra = int(target_samples * 0.4)
                for user_num in range(num_users):
                    user_id = f"sim_user_{start_user_id + user_num:04d}"
                    cluster = user_clusters[user_num]
                    user_state = {"recent_rewards": 0, "last_reward_day": -1}
                    previous_day_data = None
                    for day_num in range(days_per_user):
                        current_date = start_date + datetime.timedelta(days=day_num)
                        day_of_week = current_date.weekday()
                        month = current_date.month
                        current_daily_metrics = generate_daily_activity_metrics(cluster, day_of_week, month, previous_day_data, user_state)
                        for condition in conditions:
                            if condition["rarity"] == rarity and check_condition(current_daily_metrics, condition["parsed_condition"]):
                                if random.random() < float(condition["probability"]):
                                    label = 1
                                    base_score = int(condition["reward_score"])
                                    activity_score = compute_activity_score(current_daily_metrics)
                                    score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
                                    reward_score = random.randint(
                                        max(CONFIG_KARMA_MIN, base_score - 3),
                                        min(CONFIG_KARMA_MAX, base_score + 3 + score_range)
                                    )
                                    extra_samples.append({
                                        "user_id": user_id,
                                        "day": str(current_date),
                                        "features": {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS},
                                        "label": label,
                                        "reward_score": reward_score,
                                        "box_type": condition["box_type"],
                                        "rarity": rarity,
                                        "reason": condition["reason"],
                                        "display_reason": condition["display_reason"],
                                        "status": "delivered"
                                    })
                                    reward_counts[rarity] += 1
                                    box_type_counts[condition["box_type"]] += 1
                                    break
                        previous_day_data = current_daily_metrics
                        if len(extra_samples) >= max_extra:
                            break
                    if len(extra_samples) >= max_extra:
                        break
                all_samples.extend(extra_samples)

    # Save dataset
    with open(output_filename, "w") as f:
        json.dump(all_samples, f, indent=2)

    return all_samples, reward_counts, box_type_counts

# Main data generation
NUM_USERS_TOTAL = 1000
DAYS_PER_USER = 100
TRAIN_FRAC = 0.7  # 700 users
VAL_FRAC = 0.15  # 150 users
TEST_FRAC = 0.15  # 150 users

NUM_USERS_TRAIN = int(NUM_USERS_TOTAL * TRAIN_FRAC)
NUM_USERS_VAL = int(NUM_USERS_TOTAL * VAL_FRAC)
NUM_USERS_TEST = NUM_USERS_TOTAL - NUM_USERS_TRAIN - NUM_USERS_VAL

# Read and parse conditions.csv
conditions = []
try:
    with open("conditions.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row_num, row in enumerate(reader, 1):
            parsed_condition = parse_condition(row["condition"])
            if parsed_condition is None:
                print(f"Warning: Could not parse condition on row {row_num}: '{row['condition']}'")
            row["parsed_condition"] = parsed_condition
            conditions.append(row)
except FileNotFoundError:
    print("Error: conditions.csv not found.")
    exit()

valid_conditions = [c for c in conditions if c["parsed_condition"] is not None]
print(f"Loaded {len(valid_conditions)} valid conditions out of {len(conditions)}")

if len(valid_conditions) == 0:
    print("Error: No valid conditions found. Please check your conditions.csv file.")
    exit()

# Assign users to clusters
user_clusters = np.random.choice(list(USER_CLUSTERS.keys()), size=NUM_USERS_TOTAL, p=CLUSTER_PROBS)
train_clusters = user_clusters[:NUM_USERS_TRAIN]
val_clusters = user_clusters[NUM_USERS_TRAIN:NUM_USERS_TRAIN + NUM_USERS_VAL]
test_clusters = user_clusters[NUM_USERS_TRAIN + NUM_USERS_VAL:]

# Generate datasets
start_date = datetime.date(2024, 1, 1)

# Training dataset
train_samples, train_reward_counts, train_box_type_counts = generate_dataset(
    num_users=NUM_USERS_TRAIN,
    start_user_id=0,
    days_per_user=DAYS_PER_USER,
    output_filename="training_data.json",
    start_date=start_date,
    conditions=conditions,
    user_clusters=train_clusters
)

# Validation dataset
val_samples, val_reward_counts, val_box_type_counts = generate_dataset(
    num_users=NUM_USERS_VAL,
    start_user_id=NUM_USERS_TRAIN,
    days_per_user=DAYS_PER_USER,
    output_filename="validation_data.json",
    start_date=start_date,
    conditions=conditions,
    user_clusters=val_clusters
)

# Testing dataset
test_samples, test_reward_counts, test_box_type_counts = generate_dataset(
    num_users=NUM_USERS_TEST,
    start_user_id=NUM_USERS_TRAIN + NUM_USERS_VAL,
    days_per_user=DAYS_PER_USER,
    output_filename="testing_data.json",
    start_date=start_date,
    conditions=conditions,
    user_clusters=test_clusters
)

# Validate datasets
for samples, name in [(train_samples, "Training"), (val_samples, "Validation"), (test_samples, "Testing")]:
    issues = validate_dataset(samples, name)
    if issues:
        print(f"\n--- Realism Issues in {name} Dataset ---")
        for issue in issues[:10]:
            print(issue)
    else:
        print(f"\nNo realism issues in {name} dataset.")

# Print stats
def print_dataset_stats(samples, reward_counts, box_type_counts, name):
    num_rewarded = sum(1 for s in samples if s["label"] == 1)
    num_not_rewarded = len(samples) - num_rewarded
    print(f"\n--- {name} Dataset Stats ---")
    print(f"Total Samples: {len(samples)}")
    print(f"Number of rewarded samples (label=1): {num_rewarded} ({num_rewarded/len(samples)*100:.2f}%)")
    print(f"Number of non-rewarded samples (label=0): {num_not_rewarded} ({num_not_rewarded/len(samples)*100:.2f}%)")
    print(f"Box Type Distribution: {dict(box_type_counts)}")
    print(f"Rarity Distribution: {dict(reward_counts)}")
    if num_rewarded > 0:
        print(f"\nExample of a rewarded sample from {name}:")
        for s in samples[:10]:
            if s["label"] == 1:
                print(json.dumps(s, indent=2))
                break
    if num_not_rewarded > 0:
        print(f"\nExample of a non-rewarded sample from {name}:")
        for s in samples:
            if s["label"] == 0:
                print(json.dumps(s, indent=2))
                break

print_dataset_stats(train_samples, train_reward_counts, train_box_type_counts, "Training")
print_dataset_stats(val_samples, val_reward_counts, val_box_type_counts, "Validation")
print_dataset_stats(test_samples, test_reward_counts, test_box_type_counts, "Testing")

# Validation tests
print("\n--- Validation Tests ---")
test_metrics = {
    "login_streak": 5,
    "posts_created": 2,
    "comments_written": 3,
    "upvotes_received": 15,
    "quizzes_completed": 2,
    "buddies_messaged": 1,
    "karma_spent": 10,
    "karma_earned_today": 20
}
print(f"Test metrics: {test_metrics}")
for i, condition in enumerate(conditions[:5]):
    if condition["parsed_condition"]:
        result = check_condition(test_metrics, condition["parsed_condition"])
        print(f"Condition {i+1}: '{condition['condition']}' -> {result}")

# import csv
# import re
# import random
# import json
# import datetime
# from collections import Counter
# import numpy as np
# import pandas as pd
# from functools import lru_cache

# # Set random seed for reproducibility
# random.seed(42)
# np.random.seed(42)

# # Load config.json
# try:
#     with open("config.json", "r") as f:
#         config = json.load(f)
#     CONFIG_KARMA_MIN = config["karma_min"]
#     CONFIG_KARMA_MAX = config["karma_max"]
#     METRIC_RANGES = config["metric_ranges"]
#     USER_CLUSTERS = config["user_clusters"]
#     CLUSTER_PROBS = config["cluster_probs"]
#     TARGET_RARITY_DIST = config["target_rarity_dist"]
#     TEMPORAL_TRENDS = config["temporal_trends"]
# except FileNotFoundError:
#     print("Error: config.json not found. Exiting.")
#     exit(1)

# MODEL_FEATURE_KEYS = list(METRIC_RANGES.keys())

# # Temporal trend multipliers
# def get_temporal_multiplier(day_of_week, month):
#     # Weekend boost
#     base_mult = 1.2 if day_of_week in [5, 6] else 1.0
#     # Seasonal spike (e.g., December)
#     seasonal_mult = TEMPORAL_TRENDS.get("seasonal_multipliers", {}).get(str(month), 1.0)
#     return base_mult * seasonal_mult

# # Parse condition strings
# @lru_cache(maxsize=1000)
# def tokenize_condition(condition_str):
#     pattern = r'(\w+|>=|<=|==|<|>|\(|\)|and|or|\d+)'
#     tokens = re.findall(pattern, condition_str)
#     return [token.strip() for token in tokens if token.strip()]

# def parse_atomic_condition(tokens, start_idx):
#     if start_idx + 2 >= len(tokens):
#         return None, start_idx
#     var_token = tokens[start_idx]
#     op_token = tokens[start_idx + 1]
#     val_token = tokens[start_idx + 2]
#     if (var_token in MODEL_FEATURE_KEYS and 
#         op_token in ['>=', '<=', '==', '<', '>'] and 
#         val_token.isdigit()):
#         return (var_token, op_token, int(val_token)), start_idx + 3
#     return None, start_idx

# def parse_condition_recursive(tokens, start_idx=0):
#     left_expr, idx = parse_term(tokens, start_idx)
#     while idx < len(tokens):
#         if tokens[idx] == 'or':
#             right_expr, idx = parse_condition_recursive(tokens, idx + 1)
#             left_expr = ('or', left_expr, right_expr)
#         else:
#             break
#     return left_expr, idx

# def parse_term(tokens, start_idx):
#     left_expr, idx = parse_factor(tokens, start_idx)
#     while idx < len(tokens) and tokens[idx] == 'and':
#         right_expr, idx = parse_factor(tokens, idx + 1)
#         left_expr = ('and', left_expr, right_expr)
#     return left_expr, idx

# def parse_factor(tokens, start_idx):
#     if start_idx >= len(tokens):
#         raise ValueError("Unexpected end of expression")
#     if tokens[start_idx] == '(':
#         expr, idx = parse_condition_recursive(tokens, start_idx + 1)
#         if idx >= len(tokens) or tokens[idx] != ')':
#             raise ValueError("Missing closing parenthesis")
#         return expr, idx + 1
#     else:
#         atomic, idx = parse_atomic_condition(tokens, start_idx)
#         if atomic is None:
#             raise ValueError(f"Invalid atomic condition at position {start_idx}: {tokens[start_idx:start_idx+3]}")
#         return atomic, idx

# def parse_condition(condition_str):
#     if not condition_str.strip():
#         return None
#     tokens = tokenize_condition(condition_str)
#     if not tokens:
#         return None
#     try:
#         expr, _ = parse_condition_recursive(tokens)
#         return expr
#     except Exception as e:
#         print(f"Error parsing condition '{condition_str}': {e}")
#         return None

# def evaluate_expression(expr, metrics):
#     if isinstance(expr, tuple) and len(expr) == 3:
#         if expr[0] in MODEL_FEATURE_KEYS:
#             variable, operator, value = expr
#             metric_value = metrics.get(variable, 0)
#             if operator == '>=':
#                 return metric_value >= value
#             elif operator == '<=':
#                 return metric_value <= value
#             elif operator == '==':
#                 return metric_value == value
#             elif operator == '<':
#                 return metric_value < value
#             elif operator == '>':
#                 return metric_value > value
#             else:
#                 raise ValueError(f"Unknown operator: {operator}")
#         elif expr[0] == 'and':
#             _, left, right = expr
#             return evaluate_expression(left, metrics) and evaluate_expression(right, metrics)
#         elif expr[0] == 'or':
#             _, left, right = expr
#             return evaluate_expression(left, metrics) or evaluate_expression(right, metrics)
#     raise ValueError(f"Invalid expression: {expr}")

# def check_condition(metrics, parsed_condition):
#     if parsed_condition is None:
#         return False
#     try:
#         return evaluate_expression(parsed_condition, metrics)
#     except Exception as e:
#         print(f"Error evaluating condition: {e}")
#         return False

# def generate_daily_activity_metrics(cluster, day_of_week, month, previous_day_metrics=None, user_state=None):
#     metrics = {}
#     cluster_params = USER_CLUSTERS[cluster]
#     temporal_mult = get_temporal_multiplier(day_of_week, month)
    
#     for metric_name, (min_val, max_val) in METRIC_RANGES.items():
#         prob_active, mean_val = cluster_params.get(metric_name, (0.5, 5))
#         mean_val = mean_val * temporal_mult
#         if metric_name == "login_streak":
#             if previous_day_metrics and previous_day_metrics.get("login_streak", 0) > 0:
#                 prob_continue = prob_active * (1.2 if user_state.get("recent_rewards", 0) > 0 else 1.0)
#                 if random.random() < prob_continue:
#                     metrics[metric_name] = min(previous_day_metrics["login_streak"] + 1, max_val)
#                 else:
#                     metrics[metric_name] = 0
#             elif random.random() < prob_active:
#                 metrics[metric_name] = 1
#             else:
#                 metrics[metric_name] = 0
#         elif metric_name == "karma_earned_today":
#             if random.random() < prob_active:
#                 base_karma = (
#                     metrics.get("posts_created", 0) * 10 +
#                     metrics.get("comments_written", 0) * 5 +
#                     metrics.get("upvotes_received", 0) * 2 +
#                     metrics.get("quizzes_completed", 0) * 15
#                 )
#                 value = min(int(base_karma + np.random.exponential(mean_val * 0.5)), max_val)
#                 value = int(value * random.uniform(0.9, 1.1))
#                 if random.random() < 0.02:
#                     value = int(max_val * random.uniform(0.8, 1.0))
#                 metrics[metric_name] = max(min_val, min(value, max_val))
#             else:
#                 metrics[metric_name] = 0
#         else:
#             if random.random() < prob_active:
#                 value = min(int(np.random.exponential(mean_val)), max_val)
#                 value = int(value * random.uniform(0.9, 1.1))
#                 if random.random() < 0.02:
#                     value = int(max_val * random.uniform(0.8, 1.0))
#                 metrics[metric_name] = max(min_val, min(value, max_val))
#             else:
#                 metrics[metric_name] = 0
#         # Dependencies
#         if metric_name == "upvotes_received" and metrics.get("posts_created", 0) == 0:
#             metrics[metric_name] = 0
#         if metric_name == "shares" and metrics.get("posts_created", 0) == 0:
#             metrics[metric_name] = 0
#     return metrics

# def compute_activity_score(metrics):
#     weights = {key: 1.0 / len(METRIC_RANGES) for key in METRIC_RANGES}
#     score = 0
#     for metric, weight in weights.items():
#         max_val = METRIC_RANGES[metric][1]
#         score += weight * (metrics.get(metric, 0) / max_val if max_val > 0 else 0)
#     return score

# def determine_reward_from_metrics(daily_metrics, conditions, user_state):
#     label = 0
#     reward_score = 0
#     box_type = None
#     rarity = None
#     reason = "No reward"
#     display_reason = None

#     activity_score = compute_activity_score(daily_metrics)
#     rarity_weights = {"common": 1.0, "rare": 0.8, "epic": 0.6, "legendary": 0.4}
#     reward_freq_factor = 0.7 if user_state.get("recent_rewards", 0) >= 3 else 1.0

#     shuffled_conditions = conditions.copy()
#     random.shuffle(shuffled_conditions)

#     for condition in shuffled_conditions:
#         if condition["parsed_condition"] is None:
#             continue
#         if check_condition(daily_metrics, condition["parsed_condition"]):
#             prob = float(condition.get("probability", 1.0)) * reward_freq_factor * rarity_weights.get(condition["rarity"], 1.0)
#             if random.random() < prob:
#                 label = int(condition["label"])
#                 if label == 1:
#                     base_score = int(condition["reward_score"])
#                     score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
#                     reward_score = random.randint(
#                         max(CONFIG_KARMA_MIN, base_score - 5),
#                         min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
#                     )
#                     box_type = condition["box_type"]
#                     rarity = condition["rarity"]
#                     reason = condition["reason"]
#                     display_reason = condition["display_reason"]
#                 return label, reward_score, box_type, rarity, reason, display_reason

#     # Default mystery reward for low activity
#     is_somewhat_active = daily_metrics.get("login_streak", 0) > 0 and (
#         daily_metrics.get("karma_earned_today", 0) > 0 or
#         any(daily_metrics.get(k, 0) > 0 for k in MODEL_FEATURE_KEYS if k != "login_streak")
#     )
#     if is_somewhat_active and random.random() < 0.10:  # Increased to 10%
#         label = 1
#         reward_score = random.randint(CONFIG_KARMA_MIN, CONFIG_KARMA_MIN + 2)
#         box_type = "mystery"
#         rarity = "common"
#         reason = "Low-level activity"
#         display_reason = "Thanks for being active today!"

#     return label, reward_score, box_type, rarity, reason, display_reason

# def validate_dataset(samples):
#     issues = []
#     df = pd.DataFrame([s["features"] for s in samples])
#     for i, metrics in enumerate([s["features"] for s in samples]):
#         if metrics["upvotes_received"] > 0 and metrics["posts_created"] == 0:
#             issues.append(f"Sample {i}: Upvotes without posts")
#         if metrics["shares"] > 0 and metrics["posts_created"] == 0:
#             issues.append(f"Sample {i}: Shares without posts")
#         if metrics["karma_earned_today"] > 50 and metrics["posts_created"] == 0 and metrics["comments_written"] == 0:
#             issues.append(f"Sample {i}: High karma without posts/comments")
#     for column in df.columns:
#         skewness = df[column].skew()
#         if abs(skewness) < 0.5 and column not in ["login_streak"]:
#             issues.append(f"Metric {column} too uniform (skewness: {skewness:.2f})")
#     return issues

# # Main data generation loop
# NUM_USERS = 500
# DAYS_PER_USER = 100
# OUTPUT_FILENAME = "training_data.json"

# # Read and parse conditions.csv
# conditions = []
# try:
#     with open("conditions.csv", "r") as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row_num, row in enumerate(reader, 1):
#             parsed_condition = parse_condition(row["condition"])
#             if parsed_condition is None:
#                 print(f"Warning: Could not parse condition on row {row_num}: '{row['condition']}'")
#             row["parsed_condition"] = parsed_condition
#             conditions.append(row)
# except FileNotFoundError:
#     print("Error: conditions.csv not found.")
#     exit(1)

# valid_conditions = [c for c in conditions if c["parsed_condition"] is not None]
# print(f"Loaded {len(valid_conditions)} valid conditions out of {len(conditions)} total conditions.")

# if len(valid_conditions) == 0:
#     print("Error: No valid conditions found. Please check your conditions.csv file.")
#     exit(1)

# # Assign users to clusters
# user_clusters = np.random.choice(list(USER_CLUSTERS.keys()), size=NUM_USERS, p=CLUSTER_PROBS)

# all_training_samples = []
# start_date = datetime.date(2024, 1, 1)

# reward_counts = {rarity: 0 for rarity in TARGET_RARITY_DIST}
# box_type_counts = Counter()

# print(f"Starting data generation for {NUM_USERS} users, {DAYS_PER_USER} days each...")

# for user_num in range(NUM_USERS):
#     user_id = f"sim_user_{user_num:04d}"
#     cluster = user_clusters[user_num]
#     previous_day_data = None
#     user_state = {"recent_rewards": 0, "last_reward_day": -1}
#     if user_num % 10 == 0 and user_num > 0:
#         print(f"Processing user {user_num}/{NUM_USERS} (Cluster: {cluster})...")

#     for day_num in range(DAYS_PER_USER):
#         current_date = start_date + datetime.timedelta(days=day_num)
#         day_of_week = current_date.weekday()
#         month = current_date.month
#         current_daily_metrics = generate_daily_activity_metrics(cluster, day_of_week, month, previous_day_data, user_state)
#         label, reward_score, box_type, rarity, reason, display_reason = determine_reward_from_metrics(current_daily_metrics, conditions, user_state)
#         features_for_model = {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}

#         sample = {
#             "user_id": user_id,
#             "day": str(current_date),
#             "features": features_for_model,
#             "label": label,
#             "reward_score": reward_score,
#             "box_type": box_type,
#             "rarity": rarity,
#             "reason": reason,
#             "display_reason": display_reason
#         }
#         all_training_samples.append(sample)

#         if label == 1:
#             user_state["recent_rewards"] += 1
#             user_state["last_reward_day"] = day_num
#             reward_counts[rarity] = reward_counts.get(rarity, 0) + 1
#             box_type_counts[box_type] += 1
#         if day_num - user_state["last_reward_day"] > 5:
#             user_state["recent_rewards"] = max(0, user_state["recent_rewards"] - 1)
#         previous_day_data = current_daily_metrics

# # Balance rarity distribution
# total_rewards = sum(reward_counts.values())
# if total_rewards > 0:
#     for rarity, target_frac in TARGET_RARITY_DIST.items():
#         current_frac = reward_counts.get(rarity, 0) / total_rewards
#         if current_frac < target_frac * 0.9 and rarity in ["rare", "epic"]:
#             print(f"Balancing {rarity} rewards (current: {current_frac:.3f}, target: {target_frac:.3f})")
#             extra_samples = []
#             max_extra = int(target_frac * NUM_USERS * DAYS_PER_USER * 0.15)
#             for user_num in range(NUM_USERS):
#                 cluster = user_clusters[user_num]
#                 user_state = {"recent_rewards": 0, "last_reward_day": -1}
#                 previous_day_data = None
#                 for day_num in range(DAYS_PER_USER):
#                     current_date = start_date + datetime.timedelta(days=day_num)
#                     day_of_week = current_date.weekday()
#                     month = current_date.month
#                     current_daily_metrics = generate_daily_activity_metrics(cluster, day_of_week, month, previous_day_data, user_state)
#                     for condition in conditions:
#                         if condition["rarity"] == rarity and check_condition(current_daily_metrics, condition["parsed_condition"]):
#                             if random.random() < float(condition["probability"]):
#                                 label = 1
#                                 base_score = int(condition["reward_score"])
#                                 activity_score = compute_activity_score(current_daily_metrics)
#                                 score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
#                                 reward_score = random.randint(
#                                     max(CONFIG_KARMA_MIN, base_score - 5),
#                                     min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
#                                 )
#                                 extra_samples.append({
#                                     "user_id": f"sim_user_{user_num:04d}",
#                                     "day": str(current_date),
#                                     "features": {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS},
#                                     "label": label,
#                                     "reward_score": reward_score,
#                                     "box_type": condition["box_type"],
#                                     "rarity": rarity,
#                                     "reason": condition["reason"],
#                                     "display_reason": condition["display_reason"]
#                                 })
#                                 reward_counts[rarity] += 1
#                                 box_type_counts[condition["box_type"]] += 1
#                                 break
#                     previous_day_data = current_daily_metrics
#                     if len(extra_samples) >= max_extra:
#                         break
#                 if len(extra_samples) >= max_extra:
#                     break
#             all_training_samples.extend(extra_samples)

# # Save to JSON file
# with open(OUTPUT_FILENAME, "w") as f:
#     json.dump(all_training_samples, f, indent=2)

# # Validate dataset
# issues = validate_dataset(all_training_samples)
# if issues:
#     print("\n--- Realism Issues ---")
#     for issue in issues[:10]:
#         print(issue)
# else:
#     print("\nNo realism issues detected.")

# print(f"\nSuccessfully generated {len(all_training_samples)} training samples.")
# print(f"Data saved to: {OUTPUT_FILENAME}")

# # Dataset stats
# num_rewarded = sum(1 for s in all_training_samples if s["label"] == 1)
# num_not_rewarded = len(all_training_samples) - num_rewarded

# print(f"\n--- Dataset Stats ---")
# print(f"Total Samples: {len(all_training_samples)}")
# print(f"Number of rewarded samples (label=1): {num_rewarded} ({num_rewarded/len(all_training_samples)*100:.2f}%)")
# print(f"Number of non-rewarded samples (label=0): {num_not_rewarded} ({num_not_rewarded/len(all_training_samples)*100:.2f}%)")
# print(f"Box Type Distribution: {dict(box_type_counts)}")
# print(f"Rarity Distribution: {dict(reward_counts)}")

# if num_rewarded > 0:
#     print("\nExample of a rewarded sample:")
#     for s in all_training_samples:
#         if s["label"] == 1:
#             print(json.dumps(s, indent=2))
#             break

# if num_not_rewarded > 0:
#     print("\nExample of a non-rewarded sample:")
#     for s in all_training_samples:
#         if s["label"] == 0:
#             print(json.dumps(s, indent=2))
#             break

# # Validation tests
# print("\n--- Validation Tests ---")
# test_metrics = {
#     "login_streak": 5,
#     "posts_created": 2,
#     "comments_written": 3,
#     "upvotes_received": 15,
#     "quizzes_completed": 2,
#     "buddies_messaged": 1,
#     "karma_spent": 10,
#     "karma_earned_today": 20,
#     "shares": 1,
#     "reactions_given": 5
# }
# print(f"Test metrics: {test_metrics}")
# for i, condition in enumerate(conditions[:5]):
#     if condition["parsed_condition"]:
#         result = check_condition(test_metrics, condition["parsed_condition"])
#         print(f"Condition {i+1}: '{condition['condition']}' -> {result}")
# import csv
# import re
# import random
# import json
# import datetime
# from collections import Counter
# import numpy as np
# from functools import lru_cache

# # Set random seed for reproducibility
# random.seed(42)
# np.random.seed(42)

# # Load config.json
# try:
#     with open("config.json", "r") as f:
#         config = json.load(f)
#     CONFIG_KARMA_MIN = config["karma_min"]
#     CONFIG_KARMA_MAX = config["karma_max"]
#     METRIC_RANGES = config["metric_ranges"]
#     USER_CLUSTERS = config["user_clusters"]
#     CLUSTER_PROBS = config["cluster_probs"]
#     TARGET_RARITY_DIST = config["target_rarity_dist"]
# except FileNotFoundError:
#     print("Error: config.json not found. Exiting.")
#     exit(1)

# MODEL_FEATURE_KEYS = list(METRIC_RANGES.keys())

# # Temporal trend multipliers (e.g., higher activity on weekends)
# def get_temporal_multiplier(day_of_week):
#     # 0 = Monday, 6 = Sunday
#     if day_of_week in [5, 6]:  # Weekend
#         return 1.2
#     return 1.0

# # Parse condition strings
# @lru_cache(maxsize=1000)  # Cache parsed conditions for performance
# def tokenize_condition(condition_str):
#     pattern = r'(\w+|>=|<=|==|<|>|\(|\)|and|or|\d+)'
#     tokens = re.findall(pattern, condition_str)
#     return [token.strip() for token in tokens if token.strip()]

# def parse_atomic_condition(tokens, start_idx):
#     if start_idx + 2 >= len(tokens):
#         return None, start_idx
#     var_token = tokens[start_idx]
#     op_token = tokens[start_idx + 1]
#     val_token = tokens[start_idx + 2]
#     if (var_token in MODEL_FEATURE_KEYS and 
#         op_token in ['>=', '<=', '==', '<', '>'] and 
#         val_token.isdigit()):
#         return (var_token, op_token, int(val_token)), start_idx + 3
#     return None, start_idx

# def parse_condition_recursive(tokens, start_idx=0):
#     left_expr, idx = parse_term(tokens, start_idx)
#     while idx < len(tokens):
#         if tokens[idx] == 'or':
#             right_expr, idx = parse_condition_recursive(tokens, idx + 1)
#             left_expr = ('or', left_expr, right_expr)
#         else:
#             break
#     return left_expr, idx

# def parse_term(tokens, start_idx):
#     left_expr, idx = parse_factor(tokens, start_idx)
#     while idx < len(tokens) and tokens[idx] == 'and':
#         right_expr, idx = parse_factor(tokens, idx + 1)
#         left_expr = ('and', left_expr, right_expr)
#     return left_expr, idx

# def parse_factor(tokens, start_idx):
#     if start_idx >= len(tokens):
#         raise ValueError("Unexpected end of expression")
#     if tokens[start_idx] == '(':
#         expr, idx = parse_condition_recursive(tokens, start_idx + 1)
#         if idx >= len(tokens) or tokens[idx] != ')':
#             raise ValueError("Missing closing parenthesis")
#         return expr, idx + 1
#     else:
#         atomic, idx = parse_atomic_condition(tokens, start_idx)
#         if atomic is None:
#             raise ValueError(f"Invalid atomic condition at position {start_idx}: {tokens[start_idx:start_idx+3]}")
#         return atomic, idx

# def parse_condition(condition_str):
#     if not condition_str.strip():
#         return None
#     tokens = tokenize_condition(condition_str)
#     if not tokens:
#         return None
#     try:
#         expr, _ = parse_condition_recursive(tokens)
#         return expr
#     except Exception as e:
#         print(f"Error parsing condition '{condition_str}': {e}")
#         return None

# def evaluate_expression(expr, metrics):
#     if isinstance(expr, tuple) and len(expr) == 3:
#         if expr[0] in MODEL_FEATURE_KEYS:
#             variable, operator, value = expr
#             metric_value = metrics.get(variable, 0)
#             if operator == '>=':
#                 return metric_value >= value
#             elif operator == '<=':
#                 return metric_value <= value
#             elif operator == '==':
#                 return metric_value == value
#             elif operator == '<':
#                 return metric_value < value
#             elif operator == '>':
#                 return metric_value > value
#             else:
#                 raise ValueError(f"Unknown operator: {operator}")
#         elif expr[0] == 'and':
#             _, left, right = expr
#             return evaluate_expression(left, metrics) and evaluate_expression(right, metrics)
#         elif expr[0] == 'or':
#             _, left, right = expr
#             return evaluate_expression(left, metrics) or evaluate_expression(right, metrics)
#     raise ValueError(f"Invalid expression: {expr}")

# def check_condition(metrics, parsed_condition):
#     if parsed_condition is None:
#         return False
#     try:
#         return evaluate_expression(parsed_condition, metrics)
#     except Exception as e:
#         print(f"Error evaluating condition: {e}")
#         return False

# def generate_daily_activity_metrics(cluster, day_of_week, previous_day_metrics=None, user_state=None):
#     metrics = {}
#     cluster_params = USER_CLUSTERS[cluster]
#     temporal_mult = get_temporal_multiplier(day_of_week)
    
#     for metric_name, (min_val, max_val) in METRIC_RANGES.items():
#         prob_active, mean_val = cluster_params.get(metric_name, (0.5, 5))
#         mean_val = mean_val * temporal_mult  # Apply temporal trend
#         if metric_name == "login_streak":
#             if previous_day_metrics and previous_day_metrics.get("login_streak", 0) > 0:
#                 prob_continue = prob_active * (1.1 if user_state.get("recent_rewards", 0) > 0 else 1.0)
#                 if random.random() < prob_continue:
#                     metrics[metric_name] = min(previous_day_metrics["login_streak"] + 1, max_val)
#                 else:
#                     metrics[metric_name] = 0
#             elif random.random() < prob_active:
#                 metrics[metric_name] = 1
#             else:
#                 metrics[metric_name] = 0
#         else:
#             if random.random() < prob_active:
#                 value = min(int(np.random.exponential(mean_val)), max_val)
#                 value = int(value * random.uniform(0.9, 1.1))  # ±10% noise
#                 if random.random() < 0.02:  # Outlier chance
#                     value = int(max_val * random.uniform(0.8, 1.0))
#                 metrics[metric_name] = max(min_val, min(value, max_val))
#             else:
#                 metrics[metric_name] = 0
#         # Dependencies
#         if metric_name == "upvotes_received" and metrics.get("posts_created", 0) == 0:
#             metrics[metric_name] = 0
#         if metric_name == "shares" and metrics.get("posts_created", 0) == 0:
#             metrics[metric_name] = 0
#     return metrics

# def compute_activity_score(metrics):
#     weights = {key: 1.0 / len(METRIC_RANGES) for key in METRIC_RANGES}
#     score = 0
#     for metric, weight in weights.items():
#         max_val = METRIC_RANGES[metric][1]
#         score += weight * (metrics.get(metric, 0) / max_val if max_val > 0 else 0)
#     return score

# def determine_reward_from_metrics(daily_metrics, conditions, user_state):
#     label = 0
#     reward_score = 0
#     box_type = None
#     rarity = None
#     reason = "No reward"
#     display_reason = None

#     activity_score = compute_activity_score(daily_metrics)
#     rarity_weights = {"common": 1.0, "rare": 0.8, "epic": 0.6, "legendary": 0.5}
#     reward_freq_factor = 0.8 if user_state.get("recent_rewards", 0) >= 3 else 1.0

#     shuffled_conditions = conditions.copy()
#     random.shuffle(shuffled_conditions)

#     for condition in shuffled_conditions:
#         if condition["parsed_condition"] is None:
#             continue
#         if check_condition(daily_metrics, condition["parsed_condition"]):
#             prob = float(condition.get("probability", 1.0)) * reward_freq_factor * rarity_weights.get(condition["rarity"], 1.0)
#             if random.random() < prob:
#                 label = int(condition["label"])
#                 if label == 1:
#                     base_score = int(condition["reward_score"])
#                     score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
#                     reward_score = random.randint(
#                         max(CONFIG_KARMA_MIN, base_score - 5),
#                         min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
#                     )
#                     box_type = condition["box_type"]
#                     rarity = condition["rarity"]
#                     reason = condition["reason"]
#                     display_reason = condition["display_reason"]
#                 return label, reward_score, box_type, rarity, reason, display_reason

#     # Default mystery reward for low activity
#     is_somewhat_active = daily_metrics.get("login_streak", 0) > 0 and (
#         daily_metrics.get("karma_earned_today", 0) > 0 or
#         any(daily_metrics.get(k, 0) > 0 for k in MODEL_FEATURE_KEYS if k != "login_streak")
#     )
#     if is_somewhat_active and random.random() < 0.05:  # Increased to 5% for better balance
#         label = 1
#         reward_score = random.randint(CONFIG_KARMA_MIN, CONFIG_KARMA_MIN + 2)
#         box_type = "mystery"
#         rarity = "common"
#         reason = "Low-level activity"
#         display_reason = "Thanks for being active today!"

#     return label, reward_score, box_type, rarity, reason, display_reason

# # Main data generation loop
# NUM_USERS = 500
# DAYS_PER_USER = 100
# OUTPUT_FILENAME = "training_data.json"

# # Read and parse conditions.csv
# conditions = []
# try:
#     with open("conditions.csv", "r") as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row_num, row in enumerate(reader, 1):
#             parsed_condition = parse_condition(row["condition"])
#             if parsed_condition is None:
#                 print(f"Warning: Could not parse condition on row {row_num}: '{row['condition']}'")
#             row["parsed_condition"] = parsed_condition
#             conditions.append(row)
# except FileNotFoundError:
#     print("Error: conditions.csv not found.")
#     exit(1)

# valid_conditions = [c for c in conditions if c["parsed_condition"] is not None]
# print(f"Loaded {len(valid_conditions)} valid conditions out of {len(conditions)} total conditions.")

# if len(valid_conditions) == 0:
#     print("Error: No valid conditions found. Please check your conditions.csv file.")
#     exit(1)

# # Assign users to clusters
# user_clusters = np.random.choice(list(USER_CLUSTERS.keys()), size=NUM_USERS, p=CLUSTER_PROBS)

# all_training_samples = []
# start_date = datetime.date(2024, 1, 1)

# reward_counts = {rarity: 0 for rarity in TARGET_RARITY_DIST}
# box_type_counts = Counter()

# print(f"Starting data generation for {NUM_USERS} users, {DAYS_PER_USER} days each...")

# for user_num in range(NUM_USERS):
#     user_id = f"sim_user_{user_num:04d}"
#     cluster = user_clusters[user_num]
#     previous_day_data = None
#     user_state = {"recent_rewards": 0, "last_reward_day": -1}
#     if user_num % 10 == 0 and user_num > 0:
#         print(f"Processing user {user_num}/{NUM_USERS} (Cluster: {cluster})...")

#     for day_num in range(DAYS_PER_USER):
#         day_of_week = (start_date + datetime.timedelta(days=day_num)).weekday()
#         current_daily_metrics = generate_daily_activity_metrics(cluster, day_of_week, previous_day_data, user_state)
#         label, reward_score, box_type, rarity, reason, display_reason = determine_reward_from_metrics(current_daily_metrics, conditions, user_state)
#         features_for_model = {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}

#         sample = {
#             "user_id": user_id,
#             "day": str(start_date + datetime.timedelta(days=day_num)),
#             "features": features_for_model,
#             "label": label,
#             "reward_score": reward_score,
#             "box_type": box_type,
#             "rarity": rarity,
#             "reason": reason,
#             "display_reason": display_reason
#         }
#         all_training_samples.append(sample)

#         if label == 1:
#             user_state["recent_rewards"] += 1
#             user_state["last_reward_day"] = day_num
#             reward_counts[rarity] = reward_counts.get(rarity, 0) + 1
#             box_type_counts[box_type] += 1
#         if day_num - user_state["last_reward_day"] > 5:
#             user_state["recent_rewards"] = max(0, user_state["recent_rewards"] - 1)
#         previous_day_data = current_daily_metrics

# # Balance rarity distribution
# total_rewards = sum(reward_counts.values())
# if total_rewards > 0:
#     for rarity, target_frac in TARGET_RARITY_DIST.items():
#         current_frac = reward_counts.get(rarity, 0) / total_rewards
#         if current_frac < target_frac * 0.8:
#             print(f"Balancing {rarity} rewards (current: {current_frac:.3f}, target: {target_frac:.3f})")
#             extra_samples = []
#             for user_num in range(NUM_USERS):
#                 cluster = user_clusters[user_num]
#                 user_state = {"recent_rewards": 0, "last_reward_day": -1}
#                 previous_day_data = None
#                 for day_num in range(DAYS_PER_USER):
#                     day_of_week = (start_date + datetime.timedelta(days=day_num)).weekday()
#                     current_daily_metrics = generate_daily_activity_metrics(cluster, day_of_week, previous_day_data, user_state)
#                     for condition in conditions:
#                         if condition["rarity"] == rarity and check_condition(current_daily_metrics, condition["parsed_condition"]):
#                             if random.random() < float(condition["probability"]):
#                                 label = 1
#                                 base_score = int(condition["reward_score"])
#                                 activity_score = compute_activity_score(current_daily_metrics)
#                                 score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
#                                 reward_score = random.randint(
#                                     max(CONFIG_KARMA_MIN, base_score - 5),
#                                     min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
#                                 )
#                                 extra_samples.append({
#                                     "user_id": f"sim_user_{user_num:04d}",
#                                     "day": str(start_date + datetime.timedelta(days=day_num)),
#                                     "features": {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS},
#                                     "label": label,
#                                     "reward_score": reward_score,
#                                     "box_type": condition["box_type"],
#                                     "rarity": rarity,
#                                     "reason": condition["reason"],
#                                     "display_reason": condition["display_reason"]
#                                 })
#                                 reward_counts[rarity] += 1
#                                 box_type_counts[condition["box_type"]] += 1
#                                 break
#                     previous_day_data = current_daily_metrics
#                     if len(extra_samples) >= int(target_frac * NUM_USERS * DAYS_PER_USER * 0.1):
#                         break
#                 if len(extra_samples) >= int(target_frac * NUM_USERS * DAYS_PER_USER * 0.1):
#                     break
#             all_training_samples.extend(extra_samples)

# # Save to JSON file
# with open(OUTPUT_FILENAME, "w") as f:
#     json.dump(all_training_samples, f, indent=2)

# print(f"\nSuccessfully generated {len(all_training_samples)} training samples.")
# print(f"Data saved to: {OUTPUT_FILENAME}")

# # Dataset stats
# num_rewarded = sum(1 for s in all_training_samples if s["label"] == 1)
# num_not_rewarded = len(all_training_samples) - num_rewarded

# print(f"\n--- Dataset Stats ---")
# print(f"Total Samples: {len(all_training_samples)}")
# print(f"Number of rewarded samples (label=1): {num_rewarded} ({num_rewarded/len(all_training_samples)*100:.2f}%)")
# print(f"Number of non-rewarded samples (label=0): {num_not_rewarded} ({num_not_rewarded/len(all_training_samples)*100:.2f}%)")
# print(f"Box Type Distribution: {dict(box_type_counts)}")
# print(f"Rarity Distribution: {dict(reward_counts)}")

# if num_rewarded > 0:
#     print("\nExample of a rewarded sample:")
#     for s in all_training_samples:
#         if s["label"] == 1:
#             print(json.dumps(s, indent=2))
#             break

# if num_not_rewarded > 0:
#     print("\nExample of a non-rewarded sample:")
#     for s in all_training_samples:
#         if s["label"] == 0:
#             print(json.dumps(s, indent=2))
#             break

# # Validation tests
# print("\n--- Validation Tests ---")
# test_metrics = {
#     "login_streak": 5,
#     "posts_created": 2,
#     "comments_written": 3,
#     "upvotes_received": 15,
#     "quizzes_completed": 2,
#     "buddies_messaged": 1,
#     "karma_spent": 10,
#     "karma_earned_today": 20,
#     "shares": 1
# }
# print(f"Test metrics: {test_metrics}")
# for i, condition in enumerate(conditions[:3]):
#     if condition["parsed_condition"]:
#         result = check_condition(test_metrics, condition["parsed_condition"])
#         print(f"Condition {i+1}: '{condition['condition']}' -> {result}")
# import csv
# import re
# import random
# import json
# import datetime
# from collections import Counter
# import numpy as np
# from sklearn.mixture import GaussianMixture

# # Set random seed for reproducibility
# random.seed(42)
# np.random.seed(42)

# # Load config.json
# try:
#     with open("config.json", "r") as f:
#         config = json.load(f)
#     CONFIG_KARMA_MIN = config["karma_min"]
#     CONFIG_KARMA_MAX = config["karma_max"]
# except FileNotFoundError:
#     print("Error: config.json not found. Using default values.")
#     CONFIG_KARMA_MIN = 5
#     CONFIG_KARMA_MAX = 30

# # Define user clusters with tailored distributions
# USER_CLUSTERS = {
#     "Casual": {
#         "login_streak": (0.7, 3),  # (prob_continue, mean_streak)
#         "posts_created": (0.5, 1),
#         "comments_written": (0.6, 3),
#         "upvotes_received": (0.4, 5),
#         "quizzes_completed": (0.3, 1),
#         "buddies_messaged": (0.5, 2),
#         "karma_spent": (0.5, 10),
#         "karma_earned_today": (0.6, 15)
#     },
#     "Social": {
#         "login_streak": (0.9, 10),
#         "posts_created": (0.7, 2),
#         "comments_written": (0.8, 10),
#         "upvotes_received": (0.6, 15),
#         "quizzes_completed": (0.4, 1),
#         "buddies_messaged": (0.9, 10),
#         "karma_spent": (0.7, 30),
#         "karma_earned_today": (0.7, 25)
#     },
#     "ContentCreator": {
#         "login_streak": (0.95, 20),
#         "posts_created": (0.9, 5),
#         "comments_written": (0.7, 5),
#         "upvotes_received": (0.8, 30),
#         "quizzes_completed": (0.5, 2),
#         "buddies_messaged": (0.6, 5),
#         "karma_spent": (0.6, 20),
#         "karma_earned_today": (0.8, 40)
#     },
#     "QuizEnthusiast": {
#         "login_streak": (0.85, 8),
#         "posts_created": (0.4, 1),
#         "comments_written": (0.5, 3),
#         "upvotes_received": (0.5, 10),
#         "quizzes_completed": (0.9, 4),
#         "buddies_messaged": (0.4, 3),
#         "karma_spent": (0.5, 15),
#         "karma_earned_today": (0.6, 20)
#     }
# }
# CLUSTER_PROBS = [0.4, 0.3, 0.2, 0.1]  # Probabilities for each cluster

# # Define metric ranges (used for clipping)
# METRIC_RANGES = {
#     "login_streak": (0, 150),
#     "posts_created": (0, 10),
#     "comments_written": (0, 25),
#     "upvotes_received": (0, 100),
#     "quizzes_completed": (0, 5),
#     "buddies_messaged": (0, 30),
#     "karma_spent": (0, 200),
#     "karma_earned_today": (0, 150)
# }

# MODEL_FEATURE_KEYS = [
#     "login_streak", "posts_created", "comments_written",
#     "upvotes_received", "quizzes_completed", "buddies_messaged",
#     "karma_spent", "karma_earned_today"
# ]

# def tokenize_condition(condition_str):
#     pattern = r'(\w+|>=|<=|==|<|>|\(|\)|and|or|\d+)'
#     tokens = re.findall(pattern, condition_str)
#     return [token.strip() for token in tokens if token.strip()]

# def parse_atomic_condition(tokens, start_idx):
#     if start_idx + 2 >= len(tokens):
#         return None, start_idx
#     var_token = tokens[start_idx]
#     op_token = tokens[start_idx + 1]
#     val_token = tokens[start_idx + 2]
#     if (var_token in MODEL_FEATURE_KEYS and 
#         op_token in ['>=', '<=', '==', '<', '>'] and 
#         val_token.isdigit()):
#         return (var_token, op_token, int(val_token)), start_idx + 3
#     return None, start_idx

# def parse_condition_recursive(tokens, start_idx=0):
#     left_expr, idx = parse_term(tokens, start_idx)
#     while idx < len(tokens):
#         if tokens[idx] == 'or':
#             right_expr, idx = parse_condition_recursive(tokens, idx + 1)
#             left_expr = ('or', left_expr, right_expr)
#         else:
#             break
#     return left_expr, idx

# def parse_term(tokens, start_idx):
#     left_expr, idx = parse_factor(tokens, start_idx)
#     while idx < len(tokens) and tokens[idx] == 'and':
#         right_expr, idx = parse_factor(tokens, idx + 1)
#         left_expr = ('and', left_expr, right_expr)
#     return left_expr, idx

# def parse_factor(tokens, start_idx):
#     if start_idx >= len(tokens):
#         raise ValueError("Unexpected end of expression")
#     if tokens[start_idx] == '(':
#         expr, idx = parse_condition_recursive(tokens, start_idx + 1)
#         if idx >= len(tokens) or tokens[idx] != ')':
#             raise ValueError("Missing closing parenthesis")
#         return expr, idx + 1
#     else:
#         atomic, idx = parse_atomic_condition(tokens, start_idx)
#         if atomic is None:
#             raise ValueError(f"Invalid atomic condition at position {start_idx}: {tokens[start_idx:start_idx+3]}")
#         return atomic, idx

# def parse_condition(condition_str):
#     if not condition_str.strip():
#         return None
#     tokens = tokenize_condition(condition_str)
#     if not tokens:
#         return None
#     try:
#         expr, _ = parse_condition_recursive(tokens)
#         return expr
#     except Exception as e:
#         print(f"Error parsing condition '{condition_str}': {e}")
#         return None

# def evaluate_expression(expr, metrics):
#     if isinstance(expr, tuple) and len(expr) == 3:
#         if expr[0] in MODEL_FEATURE_KEYS:
#             variable, operator, value = expr
#             metric_value = metrics.get(variable, 0)
#             if operator == '>=':
#                 return metric_value >= value
#             elif operator == '<=':
#                 return metric_value <= value
#             elif operator == '==':
#                 return metric_value == value
#             elif operator == '<':
#                 return metric_value < value
#             elif operator == '>':
#                 return metric_value > value
#             else:
#                 raise ValueError(f"Unknown operator: {operator}")
#         elif expr[0] == 'and':
#             _, left, right = expr
#             return evaluate_expression(left, metrics) and evaluate_expression(right, metrics)
#         elif expr[0] == 'or':
#             _, left, right = expr
#             return evaluate_expression(left, metrics) or evaluate_expression(right, metrics)
#     raise ValueError(f"Invalid expression: {expr}")

# def check_condition(metrics, parsed_condition):
#     if parsed_condition is None:
#         return False
#     try:
#         return evaluate_expression(parsed_condition, metrics)
#     except Exception as e:
#         print(f"Error evaluating condition: {e}")
#         return False

# def generate_daily_activity_metrics(cluster, previous_day_metrics=None, user_state=None):
#     metrics = {}
#     for metric_name, (min_val, max_val) in METRIC_RANGES.items():
#         cluster_params = USER_CLUSTERS[cluster].get(metric_name, (0.5, 5))
#         prob_active, mean_val = cluster_params
#         if metric_name == "login_streak":
#             if previous_day_metrics and previous_day_metrics.get("login_streak", 0) > 0:
#                 prob_continue = prob_active * (1.1 if user_state.get("recent_rewards", 0) > 0 else 1.0)
#                 if random.random() < prob_continue:
#                     metrics[metric_name] = min(previous_day_metrics["login_streak"] + 1, max_val)
#                 else:
#                     metrics[metric_name] = 0
#             elif random.random() < prob_active:
#                 metrics[metric_name] = 1
#             else:
#                 metrics[metric_name] = 0
#         else:
#             if random.random() < prob_active:
#                 value = min(int(np.random.exponential(mean_val)), max_val)
#                 # Add noise (±10%)
#                 value = int(value * random.uniform(0.9, 1.1))
#                 # 2% chance of outlier
#                 if random.random() < 0.02:
#                     value = int(max_val * random.uniform(0.8, 1.0))
#                 metrics[metric_name] = max(min_val, min(value, max_val))
#             else:
#                 metrics[metric_name] = 0
#         # Dependency: upvotes require posts
#         if metric_name == "upvotes_received" and metrics.get("posts_created", 0) == 0:
#             metrics[metric_name] = 0
#     return metrics

# def compute_activity_score(metrics):
#     weights = {
#         "login_streak": 0.2, "posts_created": 0.3, "comments_written": 0.2,
#         "upvotes_received": 0.15, "quizzes_completed": 0.1, "buddies_messaged": 0.05,
#         "karma_spent": 0.05, "karma_earned_today": 0.15
#     }
#     score = 0
#     for metric, weight in weights.items():
#         max_val = METRIC_RANGES[metric][1]
#         score += weight * (metrics.get(metric, 0) / max_val if max_val > 0 else 0)
#     return score

# def determine_reward_from_metrics(daily_metrics, conditions, user_state):
#     label = 0
#     reward_score = 0
#     box_type = None
#     rarity = None

#     activity_score = compute_activity_score(daily_metrics)
#     rarity_weights = {"common": 1.0, "rare": 0.8, "epic": 0.6, "legendary": 0.5}
#     reward_freq_factor = 0.8 if user_state.get("recent_rewards", 0) >= 3 else 1.0

#     shuffled_conditions = conditions.copy()
#     random.shuffle(shuffled_conditions)

#     for condition in shuffled_conditions:
#         if condition["parsed_condition"] is None:
#             continue
#         if check_condition(daily_metrics, condition["parsed_condition"]):
#             prob = float(condition.get("probability", 1.0)) * reward_freq_factor * rarity_weights.get(condition["rarity"], 1.0)
#             if random.random() < prob:
#                 label = int(condition["label"])
#                 if label == 1:
#                     base_score = int(condition["reward_score"])
#                     # Dynamic karma adjustment based on activity
#                     score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
#                     reward_score = random.randint(
#                         max(CONFIG_KARMA_MIN, base_score - 5),
#                         min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
#                     )
#                     box_type = condition["box_type"]
#                     rarity = condition["rarity"]
#                 return label, reward_score, box_type, rarity, condition.get("reason", "Unknown")

#     # Default mystery reward for low activity
#     is_somewhat_active = daily_metrics.get("login_streak", 0) > 0 and (
#         daily_metrics.get("karma_earned_today", 0) > 0 or
#         any(daily_metrics.get(k, 0) > 0 for k in ["posts_created", "comments_written", "quizzes_completed", "buddies_messaged"])
#     )
#     if is_somewhat_active and random.random() < 0.03:
#         label = 1
#         reward_score = random.randint(CONFIG_KARMA_MIN, CONFIG_KARMA_MIN + 2)
#         box_type = "mystery"
#         rarity = "common"
#         reason = "Low-level activity"

#     return label, reward_score, box_type, rarity, reason if label == 1 else "No reward"

# # Main data generation loop
# NUM_USERS = 500
# DAYS_PER_USER = 100
# OUTPUT_FILENAME = "training_data.json"

# # Read and parse conditions.csv
# conditions = []
# try:
#     with open("conditions.csv", "r") as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row_num, row in enumerate(reader, 1):
#             parsed_condition = parse_condition(row["condition"])
#             if parsed_condition is None:
#                 print(f"Warning: Could not parse condition on row {row_num}: '{row['condition']}'")
#             row["parsed_condition"] = parsed_condition
#             conditions.append(row)
# except FileNotFoundError:
#     print("Error: conditions.csv not found.")
#     exit(1)

# valid_conditions = [c for c in conditions if c["parsed_condition"] is not None]
# print(f"Loaded {len(valid_conditions)} valid conditions out of {len(conditions)} total conditions.")

# if len(valid_conditions) == 0:
#     print("Error: No valid conditions found. Please check your conditions.csv file.")
#     exit(1)

# # Assign users to clusters
# user_clusters = np.random.choice(list(USER_CLUSTERS.keys()), size=NUM_USERS, p=CLUSTER_PROBS)

# all_training_samples = []
# start_date = datetime.date(2024, 1, 1)

# # Track reward distribution for balancing
# reward_counts = {"common": 0, "rare": 0, "epic": 0, "legendary": 0}
# box_type_counts = Counter()

# print(f"Starting data generation for {NUM_USERS} users, {DAYS_PER_USER} days each...")

# for user_num in range(NUM_USERS):
#     user_id = f"sim_user_{user_num:04d}"
#     cluster = user_clusters[user_num]
#     previous_day_data = None
#     user_state = {"recent_rewards": 0, "last_reward_day": -1}
#     if user_num % 10 == 0 and user_num > 0:
#         print(f"Processing user {user_num}/{NUM_USERS} (Cluster: {cluster})...")

#     for day_num in range(DAYS_PER_USER):
#         current_daily_metrics = generate_daily_activity_metrics(cluster, previous_day_data, user_state)
#         label, reward_score, box_type, rarity, reason = determine_reward_from_metrics(current_daily_metrics, conditions, user_state)
#         features_for_model = {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}

#         sample = {
#             "user_id": user_id,
#             "day": str(start_date + datetime.timedelta(days=day_num)),
#             "features": features_for_model,
#             "label": label,
#             "reward_score": reward_score,
#             "box_type": box_type,
#             "rarity": rarity,
#             "reason": reason
#         }
#         all_training_samples.append(sample)

#         # Update user state
#         if label == 1:
#             user_state["recent_rewards"] += 1
#             user_state["last_reward_day"] = day_num
#             reward_counts[rarity] = reward_counts.get(rarity, 0) + 1
#             box_type_counts[box_type] += 1
#         if day_num - user_state["last_reward_day"] > 5:
#             user_state["recent_rewards"] = max(0, user_state["recent_rewards"] - 1)
#         previous_day_data = current_daily_metrics

# # Balance rare classes
# target_rarity_dist = {"common": 0.6, "rare": 0.25, "epic": 0.1, "legendary": 0.05}
# total_rewards = sum(reward_counts.values())
# if total_rewards > 0:
#     for rarity, target_frac in target_rarity_dist.items():
#         current_frac = reward_counts.get(rarity, 0) / total_rewards
#         if current_frac < target_frac * 0.8:
#             print(f"Balancing {rarity} rewards (current: {current_frac:.3f}, target: {target_frac:.3f})")
#             extra_samples = []
#             for user_num in range(NUM_USERS):
#                 cluster = user_clusters[user_num]
#                 user_state = {"recent_rewards": 0, "last_reward_day": -1}
#                 previous_day_data = None
#                 for day_num in range(DAYS_PER_USER):
#                     current_daily_metrics = generate_daily_activity_metrics(cluster, previous_day_data, user_state)
#                     for condition in conditions:
#                         if condition["rarity"] == rarity and check_condition(current_daily_metrics, condition["parsed_condition"]):
#                             if random.random() < float(condition["probability"]):
#                                 label = 1
#                                 base_score = int(condition["reward_score"])
#                                 activity_score = compute_activity_score(current_daily_metrics)
#                                 score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
#                                 reward_score = random.randint(
#                                     max(CONFIG_KARMA_MIN, base_score - 5),
#                                     min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
#                                 )
#                                 extra_samples.append({
#                                     "user_id": f"sim_user_{user_num:04d}",
#                                     "day": str(start_date + datetime.timedelta(days=day_num)),
#                                     "features": {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS},
#                                     "label": label,
#                                     "reward_score": reward_score,
#                                     "box_type": condition["box_type"],
#                                     "rarity": rarity,
#                                     "reason": condition["reason"]
#                                 })
#                                 reward_counts[rarity] += 1
#                                 box_type_counts[condition["box_type"]] += 1
#                                 break
#                     previous_day_data = current_daily_metrics
#                     if len(extra_samples) >= int(target_frac * NUM_USERS * DAYS_PER_USER * 0.1):
#                         break
#                 if len(extra_samples) >= int(target_frac * NUM_USERS * DAYS_PER_USER * 0.1):
#                     break
#             all_training_samples.extend(extra_samples)

# # Save to JSON file
# with open(OUTPUT_FILENAME, "w") as f:
#     json.dump(all_training_samples, f, indent=2)

# print(f"\nSuccessfully generated {len(all_training_samples)} training samples.")
# print(f"Data saved to: {OUTPUT_FILENAME}")

# # Dataset stats
# num_rewarded = sum(1 for s in all_training_samples if s["label"] == 1)
# num_not_rewarded = len(all_training_samples) - num_rewarded

# print(f"\n--- Dataset Stats ---")
# print(f"Total Samples: {len(all_training_samples)}")
# print(f"Number of rewarded samples (label=1): {num_rewarded} ({num_rewarded/len(all_training_samples)*100:.2f}%)")
# print(f"Number of non-rewarded samples (label=0): {num_not_rewarded} ({num_not_rewarded/len(all_training_samples)*100:.2f}%)")
# print(f"Box Type Distribution: {dict(box_type_counts)}")
# print(f"Rarity Distribution: {dict(reward_counts)}")

# if num_rewarded > 0:
#     print("\nExample of a rewarded sample:")
#     for s in all_training_samples:
#         if s["label"] == 1:
#             print(json.dumps(s, indent=2))
#             break

# if num_not_rewarded > 0:
#     print("\nExample of a non-rewarded sample:")
#     for s in all_training_samples:
#         if s["label"] == 0:
#             print(json.dumps(s, indent=2))
#             break

# # Condition testing
# print("\n--- Condition Testing ---")
# test_metrics = {
#     "login_streak": 5,
#     "posts_created": 2,
#     "comments_written": 3,
#     "upvotes_received": 15,
#     "quizzes_completed": 2,
#     "buddies_messaged": 1,
#     "karma_spent": 10,
#     "karma_earned_today": 20
# }
# print(f"Test metrics: {test_metrics}")
# for i, condition in enumerate(conditions[:3]):
#     if condition["parsed_condition"]:
#         result = check_condition(test_metrics, condition["parsed_condition"])
#         print(f"Condition {i+1}: '{condition['condition']}' -> {result}")
# # import csv
# # import re
# # import random
# # import json
# # import datetime
# # from collections import Counter

# # # Set random seed for reproducibility
# # random.seed(42)

# # # Load config.json
# # try:
# #     with open("config.json", "r") as f:
# #         config = json.load(f)
# #     CONFIG_KARMA_MIN = config["karma_min"]  # e.g., 5
# #     CONFIG_KARMA_MAX = config["karma_max"]  # e.g., 30
# # except FileNotFoundError:
# #     print("Error: config.json not found. Using default values.")
# #     CONFIG_KARMA_MIN = 5
# #     CONFIG_KARMA_MAX = 30

# # # Define realistic ranges for each user metric
# # METRIC_RANGES = {
# #     "login_streak": (0, 150),
# #     "posts_created": (0, 10),
# #     "comments_written": (0, 25),
# #     "upvotes_received": (0, 100),
# #     "quizzes_completed": (0, 5),
# #     "buddies_messaged": (0, 30),
# #     "karma_spent": (0, 200),
# #     "karma_earned_today": (0, 150)
# # }

# # # Model feature keys
# # MODEL_FEATURE_KEYS = [
# #     "login_streak", "posts_created", "comments_written",
# #     "upvotes_received", "quizzes_completed", "buddies_messaged",
# #     "karma_spent", "karma_earned_today"
# # ]

# # def tokenize_condition(condition_str):
# #     """
# #     Tokenize a condition string into individual tokens.
# #     Handles operators, parentheses, variables, numbers, and logical operators.
# #     """
# #     # Pattern to match: variables, operators, numbers, parentheses, and/or
# #     pattern = r'(\w+|>=|<=|==|<|>|\(|\)|and|or|\d+)'
# #     tokens = re.findall(pattern, condition_str)
# #     return [token.strip() for token in tokens if token.strip()]

# # def parse_atomic_condition(tokens, start_idx):
# #     """
# #     Parse an atomic condition like 'variable >= value' starting at start_idx.
# #     Returns (constraint_tuple, next_index) or (None, next_index) if not a valid atomic condition.
# #     """
# #     if start_idx + 2 >= len(tokens):
# #         return None, start_idx
    
# #     var_token = tokens[start_idx]
# #     op_token = tokens[start_idx + 1]
# #     val_token = tokens[start_idx + 2]
    
# #     # Check if this looks like a valid atomic condition
# #     if (var_token in MODEL_FEATURE_KEYS and 
# #         op_token in ['>=', '<=', '==', '<', '>'] and 
# #         val_token.isdigit()):
# #         return (var_token, op_token, int(val_token)), start_idx + 3
    
# #     return None, start_idx

# # def parse_condition_recursive(tokens, start_idx=0):
# #     """
# #     Recursively parse a condition with proper operator precedence.
# #     'and' has higher precedence than 'or'.
# #     Returns (parsed_expression, next_index)
# #     """
# #     # Parse the first term (could be atomic condition or parenthesized expression)
# #     left_expr, idx = parse_term(tokens, start_idx)
    
# #     while idx < len(tokens):
# #         if tokens[idx] == 'or':
# #             # OR has lower precedence, so we parse the right side
# #             right_expr, idx = parse_condition_recursive(tokens, idx + 1)
# #             left_expr = ('or', left_expr, right_expr)
# #         else:
# #             break
    
# #     return left_expr, idx

# # def parse_term(tokens, start_idx):
# #     """
# #     Parse a term (handles 'and' operations and atomic expressions).
# #     """
# #     left_expr, idx = parse_factor(tokens, start_idx)
    
# #     while idx < len(tokens) and tokens[idx] == 'and':
# #         right_expr, idx = parse_factor(tokens, idx + 1)
# #         left_expr = ('and', left_expr, right_expr)
    
# #     return left_expr, idx

# # def parse_factor(tokens, start_idx):
# #     """
# #     Parse a factor (atomic condition or parenthesized expression).
# #     """
# #     if start_idx >= len(tokens):
# #         raise ValueError("Unexpected end of expression")
    
# #     if tokens[start_idx] == '(':
# #         # Parse parenthesized expression
# #         expr, idx = parse_condition_recursive(tokens, start_idx + 1)
# #         if idx >= len(tokens) or tokens[idx] != ')':
# #             raise ValueError("Missing closing parenthesis")
# #         return expr, idx + 1
# #     else:
# #         # Try to parse atomic condition
# #         atomic, idx = parse_atomic_condition(tokens, start_idx)
# #         if atomic is None:
# #             raise ValueError(f"Invalid atomic condition at position {start_idx}: {tokens[start_idx:start_idx+3]}")
# #         return atomic, idx

# # def parse_condition(condition_str):
# #     """
# #     Parse a condition string into a structured expression tree.
# #     """
# #     if not condition_str.strip():
# #         return None
    
# #     tokens = tokenize_condition(condition_str)
# #     if not tokens:
# #         return None
    
# #     try:
# #         expr, _ = parse_condition_recursive(tokens)
# #         return expr
# #     except Exception as e:
# #         print(f"Error parsing condition '{condition_str}': {e}")
# #         print(f"Tokens: {tokens}")
# #         return None

# # def evaluate_expression(expr, metrics):
# #     """
# #     Evaluate a parsed expression tree against metrics.
# #     """
# #     if isinstance(expr, tuple) and len(expr) == 3:
# #         if expr[0] in MODEL_FEATURE_KEYS:
# #             # This is an atomic condition (variable, operator, value)
# #             variable, operator, value = expr
# #             metric_value = metrics.get(variable, 0)
            
# #             if operator == '>=':
# #                 return metric_value >= value
# #             elif operator == '<=':
# #                 return metric_value <= value
# #             elif operator == '==':
# #                 return metric_value == value
# #             elif operator == '<':
# #                 return metric_value < value
# #             elif operator == '>':
# #                 return metric_value > value
# #             else:
# #                 raise ValueError(f"Unknown operator: {operator}")
        
# #         elif expr[0] == 'and':
# #             # Logical AND
# #             _, left, right = expr
# #             return evaluate_expression(left, metrics) and evaluate_expression(right, metrics)
        
# #         elif expr[0] == 'or':
# #             # Logical OR
# #             _, left, right = expr
# #             return evaluate_expression(left, metrics) or evaluate_expression(right, metrics)
    
# #     raise ValueError(f"Invalid expression: {expr}")

# # def check_condition(metrics, parsed_condition):
# #     """
# #     Check if metrics satisfy the parsed condition.
# #     """
# #     if parsed_condition is None:
# #         return False
    
# #     try:
# #         return evaluate_expression(parsed_condition, metrics)
# #     except Exception as e:
# #         print(f"Error evaluating condition: {e}")
# #         return False

# # def generate_daily_activity_metrics(previous_day_metrics=None):
# #     """
# #     Generate a plausible set of daily metrics for one day with realistic patterns.
# #     Uses exponential distributions and streak continuity.
# #     """
# #     metrics = {}
# #     for metric_name, (min_val, max_val) in METRIC_RANGES.items():
# #         if metric_name == "login_streak":
# #             if previous_day_metrics and previous_day_metrics.get("login_streak", 0) > 0:
# #                 if random.random() < 0.95:  # 95% chance to continue streak
# #                     metrics[metric_name] = min(previous_day_metrics["login_streak"] + 1, max_val)
# #                 else:
# #                     metrics[metric_name] = 0
# #             elif random.random() < 0.3:  # 30% chance to start a streak
# #                 metrics[metric_name] = 1
# #             else:
# #                 metrics[metric_name] = 0
# #         elif metric_name in ["posts_created", "quizzes_completed", "comments_written"]:
# #             metrics[metric_name] = min(int(random.expovariate(0.8)), max_val)
# #         elif metric_name == "upvotes_received":
# #             if metrics.get("posts_created", 0) > 0:  # Upvotes only if posts created
# #                 metrics[metric_name] = min(int(random.expovariate(0.1)), max_val)
# #             else:
# #                 metrics[metric_name] = 0
# #         elif metric_name == "buddies_messaged":
# #             metrics[metric_name] = min(int(random.expovariate(0.5)), max_val)
# #         else:  # karma_spent, karma_earned_today
# #             metrics[metric_name] = random.randint(min_val, max_val)
# #     return metrics

# # def determine_reward_from_metrics(daily_metrics, conditions):
# #     """
# #     Determine if a reward is given based on conditions from conditions.csv.
# #     Includes probabilistic triggers and default mystery reward.
# #     """
# #     label = 0
# #     reward_score = 0
# #     box_type = None
# #     rarity = None

# #     # Shuffle conditions to avoid bias towards earlier conditions
# #     shuffled_conditions = conditions.copy()
# #     random.shuffle(shuffled_conditions)

# #     for condition in shuffled_conditions:
# #         if condition["parsed_condition"] is None:
# #             continue
            
# #         if check_condition(daily_metrics, condition["parsed_condition"]):
# #             prob = float(condition.get("probability", 1.0))
# #             if random.random() < prob:
# #                 label = int(condition["label"])
# #                 if label == 1:
# #                     reward_score = random.randint(
# #                         max(CONFIG_KARMA_MIN, int(condition["reward_score"]) - 5),
# #                         min(CONFIG_KARMA_MAX, int(condition["reward_score"]) + 5)
# #                     )
# #                     box_type = condition["box_type"]
# #                     rarity = condition["rarity"]
# #                 else:
# #                     reward_score = 0
# #                     box_type = None
# #                     rarity = None
# #                 return label, reward_score, box_type, rarity

# #     # Default mystery reward for low-level activity
# #     is_somewhat_active = daily_metrics.get("login_streak", 0) > 0 and (
# #         daily_metrics.get("karma_earned_today", 0) > 0 or
# #         any(daily_metrics.get(k, 0) > 0 for k in ["posts_created", "comments_written", "quizzes_completed", "buddies_messaged"])
# #     )
# #     if is_somewhat_active and random.random() < 0.03:  # 3% chance
# #         label = 1
# #         reward_score = random.randint(CONFIG_KARMA_MIN, CONFIG_KARMA_MIN + 2)
# #         box_type = "mystery"
# #         rarity = "common"

# #     return label, reward_score, box_type, rarity

# # # Main data generation loop
# # NUM_USERS = 500
# # DAYS_PER_USER = 100
# # OUTPUT_FILENAME = "training_data.json"

# # # Read and parse conditions.csv
# # conditions = []
# # try:
# #     with open("conditions.csv", "r") as csvfile:
# #         reader = csv.DictReader(csvfile)
# #         for row_num, row in enumerate(reader, 1):
# #             # Parse the condition string
# #             parsed_condition = parse_condition(row["condition"])
# #             if parsed_condition is None:
# #                 print(f"Warning: Could not parse condition on row {row_num}: '{row['condition']}'")
            
# #             row["parsed_condition"] = parsed_condition
# #             conditions.append(row)
            
# # except FileNotFoundError:
# #     print("Error: conditions.csv not found.")
# #     exit(1)

# # # Validate that we have some valid conditions
# # valid_conditions = [c for c in conditions if c["parsed_condition"] is not None]
# # print(f"Loaded {len(valid_conditions)} valid conditions out of {len(conditions)} total conditions.")

# # if len(valid_conditions) == 0:
# #     print("Error: No valid conditions found. Please check your conditions.csv file.")
# #     exit(1)

# # all_training_samples = []
# # start_date = datetime.date(2024, 1, 1)

# # print(f"Starting data generation for {NUM_USERS} users, {DAYS_PER_USER} days each...")
# # print(f"Total samples to generate: {NUM_USERS * DAYS_PER_USER}")

# # for user_num in range(NUM_USERS):
# #     user_id = f"sim_user_{user_num:04d}"
# #     previous_day_data = None
# #     if user_num % 10 == 0 and user_num > 0:
# #         print(f"Processing user {user_num}/{NUM_USERS}...")

# #     for day_num in range(DAYS_PER_USER):
# #         current_daily_metrics = generate_daily_activity_metrics(previous_day_data)
# #         label, reward_score, box_type, rarity = determine_reward_from_metrics(current_daily_metrics, conditions)
# #         features_for_model = {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}

# #         sample = {
# #             "features": features_for_model,
# #             "label": label,
# #             "reward_score": reward_score if label == 1 else 0,
# #             "box_type": box_type if label == 1 else None,
# #             "rarity": rarity if label == 1 else None
# #         }
# #         all_training_samples.append(sample)
# #         previous_day_data = current_daily_metrics

# # # Save to JSON file
# # with open(OUTPUT_FILENAME, "w") as f:
# #     json.dump(all_training_samples, f, indent=2)

# # print(f"\nSuccessfully generated {len(all_training_samples)} training samples.")
# # print(f"Data saved to: {OUTPUT_FILENAME}")

# # # Sanity check
# # num_rewarded = sum(1 for s in all_training_samples if s["label"] == 1)
# # num_not_rewarded = len(all_training_samples) - num_rewarded

# # print(f"\n--- Dataset Stats ---")
# # print(f"Total Samples: {len(all_training_samples)}")
# # print(f"Number of rewarded samples (label=1): {num_rewarded} ({num_rewarded/len(all_training_samples)*100:.2f}%)")
# # print(f"Number of non-rewarded samples (label=0): {num_not_rewarded} ({num_not_rewarded/len(all_training_samples)*100:.2f}%)")

# # if num_rewarded > 0:
# #     print("\n--- Rewarded Sample Breakdown ---")
# #     rewarded_box_types = [s['box_type'] for s in all_training_samples if s['label'] == 1]
# #     rewarded_rarities = [s['rarity'] for s in all_training_samples if s['label'] == 1]
# #     print(f"Box Type Distribution: {Counter(rewarded_box_types)}")
# #     print(f"Rarity Distribution: {Counter(rewarded_rarities)}")
    
# #     print("\nExample of a rewarded sample:")
# #     for s in all_training_samples:
# #         if s["label"] == 1:
# #             print(json.dumps(s, indent=2))
# #             break

# # if num_not_rewarded > 0:
# #     print("\nExample of a non-rewarded sample:")
# #     for s in all_training_samples:
# #         if s["label"] == 0:
# #             print(json.dumps(s, indent=2))
# #             break

# # # Test some conditions manually for debugging
# # print("\n--- Condition Testing ---")
# # test_metrics = {
# #     "login_streak": 5,
# #     "posts_created": 2,
# #     "comments_written": 3,
# #     "upvotes_received": 15,
# #     "quizzes_completed": 2,
# #     "buddies_messaged": 1,
# #     "karma_spent": 10,
# #     "karma_earned_today": 20
# # }

# # print(f"Test metrics: {test_metrics}")
# # for i, condition in enumerate(conditions[:3]):  # Test first 3 conditions
# #     if condition["parsed_condition"]:
# #         result = check_condition(test_metrics, condition["parsed_condition"])
# #         print(f"Condition {i+1}: '{condition['condition']}' -> {result}")