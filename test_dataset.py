import random
import csv
import json
import datetime
from collections import Counter
import numpy as np
from dataset import parse_condition, check_condition, MODEL_FEATURE_KEYS, METRIC_RANGES, USER_CLUSTERS, CLUSTER_PROBS

# Set random seed for reproducibility (different from training to ensure variation)
random.seed(123)
np.random.seed(123)

# Load config.json
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    CONFIG_KARMA_MIN = config["karma_min"]
    CONFIG_KARMA_MAX = config["karma_max"]
except FileNotFoundError:
    print("Error: config.json not found. Using default values.")
    CONFIG_KARMA_MIN = 5
    CONFIG_KARMA_MAX = 30

# Load and parse conditions.csv
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
    exit(1)

valid_conditions = [c for c in conditions if c["parsed_condition"] is not None]
print(f"Loaded {len(valid_conditions)} valid conditions out of {len(conditions)} total conditions.")

if len(valid_conditions) == 0:
    print("Error: No valid conditions found. Please check your conditions.csv file.")
    exit(1)

# Define functions from dataset.py (if not imported)
def compute_activity_score(metrics):
    weights = {
        "login_streak": 0.2, "posts_created": 0.3, "comments_written": 0.2,
        "upvotes_received": 0.15, "quizzes_completed": 0.1, "buddies_messaged": 0.05,
        "karma_spent": 0.05, "karma_earned_today": 0.15
    }
    score = 0
    for metric, weight in weights.items():
        max_val = METRIC_RANGES[metric][1]
        score += weight * (metrics.get(metric, 0) / max_val if max_val > 0 else 0)
    return score

def generate_daily_activity_metrics(cluster, previous_day_metrics=None, user_state=None):
    metrics = {}
    for metric_name, (min_val, max_val) in METRIC_RANGES.items():
        cluster_params = USER_CLUSTERS[cluster].get(metric_name, (0.5, 5))
        prob_active, mean_val = cluster_params
        if metric_name == "login_streak":
            if previous_day_metrics and previous_day_metrics.get("login_streak", 0) > 0:
                prob_continue = prob_active * (1.1 if user_state.get("recent_rewards", 0) > 0 else 1.0)
                if random.random() < prob_continue:
                    metrics[metric_name] = min(previous_day_metrics["login_streak"] + 1, max_val)
                else:
                    metrics[metric_name] = 0
            elif random.random() < prob_active:
                metrics[metric_name] = 1
            else:
                metrics[metric_name] = 0
        else:
            if random.random() < prob_active:
                value = min(int(np.random.exponential(mean_val)), max_val)
                value = int(value * random.uniform(0.9, 1.1))  # ±10% noise
                if random.random() < 0.02:  # 2% chance of outlier
                    value = int(max_val * random.uniform(0.8, 1.0))
                metrics[metric_name] = max(min_val, min(value, max_val))
            else:
                metrics[metric_name] = 0
        if metric_name == "upvotes_received" and metrics.get("posts_created", 0) == 0:
            metrics[metric_name] = 0
    return metrics

def determine_reward_from_metrics(daily_metrics, conditions, user_state):
    label = 0
    reward_score = 0
    box_type = None
    rarity = None
    reason = "No reward"

    activity_score = compute_activity_score(daily_metrics)
    rarity_weights = {"common": 1.0, "rare": 0.8, "epic": 0.6, "legendary": 0.5}
    reward_freq_factor = 0.8 if user_state.get("recent_rewards", 0) >= 3 else 1.0

    shuffled_conditions = conditions.copy()
    random.shuffle(shuffled_conditions)

    for condition in shuffled_conditions:
        if condition["parsed_condition"] is None:
            continue
        if check_condition(daily_metrics, condition["parsed_condition"]):
            prob = float(condition.get("probability", 1.0)) * reward_freq_factor * rarity_weights.get(condition["rarity"], 1.0)
            if random.random() < prob:
                label = int(condition["label"])
                if label == 1:
                    base_score = int(condition["reward_score"])
                    score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
                    reward_score = random.randint(
                        max(CONFIG_KARMA_MIN, base_score - 5),
                        min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
                    )
                    box_type = condition["box_type"]
                    rarity = condition["rarity"]
                    reason = condition.get("reason", "Unknown")
                return label, reward_score, box_type, rarity, reason

    is_somewhat_active = daily_metrics.get("login_streak", 0) > 0 and (
        daily_metrics.get("karma_earned_today", 0) > 0 or
        any(daily_metrics.get(k, 0) > 0 for k in ["posts_created", "comments_written", "quizzes_completed", "buddies_messaged"])
    )
    if is_somewhat_active and random.random() < 0.03:
        label = 1
        reward_score = random.randint(CONFIG_KARMA_MIN, CONFIG_KARMA_MIN + 2)
        box_type = "mystery"
        rarity = "common"
        reason = "Low-level activity"

    return label, reward_score, box_type, rarity, reason

# Data generation parameters
NUM_USERS = 100  # Reduced for testing
DAYS_PER_USER = 50  # Reduced for testing
OUTPUT_FILENAME = "testing_data.json"

# Assign users to clusters
user_clusters = np.random.choice(list(USER_CLUSTERS.keys()), size=NUM_USERS, p=CLUSTER_PROBS)

# Main data generation loop
all_testing_samples = []
start_date = datetime.date(2024, 1, 1)
reward_counts = {"common": 0, "rare": 0, "epic": 0, "legendary": 0}
box_type_counts = Counter()

print(f"Starting test data generation for {NUM_USERS} users, {DAYS_PER_USER} days each...")
print(f"Total samples to generate: {NUM_USERS * DAYS_PER_USER}")

for user_num in range(NUM_USERS):
    user_id = f"test_user_{user_num:04d}"
    cluster = user_clusters[user_num]
    previous_day_data = None
    user_state = {"recent_rewards": 0, "last_reward_day": -1}
    if user_num % 10 == 0 and user_num > 0:
        print(f"Processing user {user_num}/{NUM_USERS} (Cluster: {cluster})...")

    for day_num in range(DAYS_PER_USER):
        current_daily_metrics = generate_daily_activity_metrics(cluster, previous_day_data, user_state)
        label, reward_score, box_type, rarity, reason = determine_reward_from_metrics(current_daily_metrics, conditions, user_state)
        features_for_model = {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}

        sample = {
            "user_id": user_id,
            "day": str(start_date + datetime.timedelta(days=day_num)),
            "features": features_for_model,
            "label": label,
            "reward_score": reward_score,
            "box_type": box_type,
            "rarity": rarity,
            "reason": reason
        }
        all_testing_samples.append(sample)

        if label == 1:
            user_state["recent_rewards"] += 1
            user_state["last_reward_day"] = day_num
            reward_counts[rarity] = reward_counts.get(rarity, 0) + 1
            box_type_counts[box_type] += 1
        if day_num - user_state["last_reward_day"] > 5:
            user_state["recent_rewards"] = max(0, user_state["recent_rewards"] - 1)
        previous_day_data = current_daily_metrics

# Light balancing for rare cases (less aggressive than training)
target_rarity_dist = {"common": 0.6, "rare": 0.25, "epic": 0.1, "legendary": 0.05}
total_rewards = sum(reward_counts.values())
if total_rewards > 0:
    for rarity, target_frac in target_rarity_dist.items():
        current_frac = reward_counts.get(rarity, 0) / total_rewards
        if current_frac < target_frac * 0.7:  # Allow more tolerance than training
            print(f"Balancing {rarity} rewards (current: {current_frac:.3f}, target: {target_frac:.3f})")
            extra_samples = []
            for user_num in range(NUM_USERS):
                cluster = user_clusters[user_num]
                user_state = {"recent_rewards": 0, "last_reward_day": -1}
                previous_day_data = None
                for day_num in range(DAYS_PER_USER):
                    current_daily_metrics = generate_daily_activity_metrics(cluster, previous_day_data, user_state)
                    for condition in conditions:
                        if condition["rarity"] == rarity and check_condition(current_daily_metrics, condition["parsed_condition"]):
                            if random.random() < float(condition["probability"]):
                                label = 1
                                base_score = int(condition["reward_score"])
                                activity_score = compute_activity_score(current_daily_metrics)
                                score_range = max(0, int(activity_score * (CONFIG_KARMA_MAX - CONFIG_KARMA_MIN)))
                                reward_score = random.randint(
                                    max(CONFIG_KARMA_MIN, base_score - 5),
                                    min(CONFIG_KARMA_MAX, base_score + 5 + score_range)
                                )
                                extra_samples.append({
                                    "user_id": f"test_user_{user_num:04d}",
                                    "day": str(start_date + datetime.timedelta(days=day_num)),
                                    "features": {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS},
                                    "label": label,
                                    "reward_score": reward_score,
                                    "box_type": condition["box_type"],
                                    "rarity": rarity,
                                    "reason": condition["reason"]
                                })
                                reward_counts[rarity] += 1
                                box_type_counts[condition["box_type"]] += 1
                                break
                    previous_day_data = current_daily_metrics
                    if len(extra_samples) >= int(target_frac * NUM_USERS * DAYS_PER_USER * 0.05):  # Lighter balancing
                        break
                if len(extra_samples) >= int(target_frac * NUM_USERS * DAYS_PER_USER * 0.05):
                    break
            all_testing_samples.extend(extra_samples)

# Save to JSON file
with open(OUTPUT_FILENAME, "w") as f:
    json.dump(all_testing_samples, f, indent=2)

print(f"\nSuccessfully generated {len(all_testing_samples)} testing samples.")
print(f"Data saved to: {OUTPUT_FILENAME}")

# Dataset stats
num_rewarded = sum(1 for s in all_testing_samples if s["label"] == 1)
num_not_rewarded = len(all_testing_samples) - num_rewarded

print(f"\n--- Testing Dataset Stats ---")
print(f"Total Samples: {len(all_testing_samples)}")
print(f"Number of rewarded samples (label=1): {num_rewarded} ({num_rewarded/len(all_testing_samples)*100:.2f}%)")
print(f"Number of non-rewarded samples (label=0): {num_not_rewarded} ({num_not_rewarded/len(all_testing_samples)*100:.2f}%)")
print(f"Box Type Distribution: {dict(box_type_counts)}")
print(f"Rarity Distribution: {dict(reward_counts)}")

if num_rewarded > 0:
    print("\nExample of a rewarded sample:")
    for s in all_testing_samples:
        if s["label"] == 1:
            print(json.dumps(s, indent=2))
            break

if num_not_rewarded > 0:
    print("\nExample of a non-rewarded sample:")
    for s in all_testing_samples:
        if s["label"] == 0:
            print(json.dumps(s, indent=2))
            break

# Condition testing
print("\n--- Condition Testing ---")
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
for i, condition in enumerate(conditions[:3]):
    if condition["parsed_condition"]:
        result = check_condition(test_metrics, condition["parsed_condition"])
        print(f"Condition {i+1}: '{condition['condition']}' -> {result}")
# import random
# import json
# from dataset import generate_daily_activity_metrics, determine_reward_from_metrics, MODEL_FEATURE_KEYS, conditions

# # Set random seed for reproducibility
# random.seed(123)

# # Data generation parameters
# NUM_USERS = 500
# DAYS_PER_USER = 100
# OUTPUT_FILENAME = "testing_data.json"

# # Main data generation loop
# all_testing_samples = []
# print(f"Starting test data generation for {NUM_USERS} users, {DAYS_PER_USER} days each...")
# print(f"Total samples to generate: {NUM_USERS * DAYS_PER_USER}")

# for user_num in range(NUM_USERS):
#     user_id = f"test_user_{user_num:04d}"
#     previous_day_data = None
#     if user_num % 10 == 0 and user_num > 0:
#         print(f"Processing user {user_num}/{NUM_USERS}...")

#     for day_num in range(DAYS_PER_USER):
#         current_daily_metrics = generate_daily_activity_metrics(previous_day_data)
#         label, reward_score, box_type, rarity = determine_reward_from_metrics(current_daily_metrics, conditions)
#         features_for_model = {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}

#         sample = {
#             "features": features_for_model,
#             "label": label,
#             "reward_score": reward_score if label == 1 else 0,
#             "box_type": box_type if label == 1 else None,
#             "rarity": rarity if label == 1 else None
#         }
#         all_testing_samples.append(sample)
#         previous_day_data = current_daily_metrics

# # Save to JSON file
# with open(OUTPUT_FILENAME, "w") as f:
#     json.dump(all_testing_samples, f, indent=2)

# print(f"\nSuccessfully generated {len(all_testing_samples)} testing samples.")
# print(f"Data saved to: {OUTPUT_FILENAME}")