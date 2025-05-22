import random
import json
from dataset import generate_daily_activity_metrics, determine_reward_from_metrics, MODEL_FEATURE_KEYS, conditions

# Set random seed for reproducibility
random.seed(42)

# Data generation parameters
NUM_USERS = 100
DAYS_PER_USER = 100
OUTPUT_FILENAME = "testing_data.json"

# Main data generation loop
all_testing_samples = []
print(f"Starting test data generation for {NUM_USERS} users, {DAYS_PER_USER} days each...")
print(f"Total samples to generate: {NUM_USERS * DAYS_PER_USER}")

for user_num in range(NUM_USERS):
    user_id = f"test_user_{user_num:04d}"
    previous_day_data = None
    if user_num % 10 == 0 and user_num > 0:
        print(f"Processing user {user_num}/{NUM_USERS}...")

    for day_num in range(DAYS_PER_USER):
        current_daily_metrics = generate_daily_activity_metrics(previous_day_data)
        label, reward_score, box_type, rarity = determine_reward_from_metrics(current_daily_metrics, conditions)
        features_for_model = {key: current_daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}

        sample = {
            "features": features_for_model,
            "label": label,
            "reward_score": reward_score if label == 1 else 0,
            "box_type": box_type if label == 1 else None,
            "rarity": rarity if label == 1 else None
        }
        all_testing_samples.append(sample)
        previous_day_data = current_daily_metrics

# Save to JSON file
with open(OUTPUT_FILENAME, "w") as f:
    json.dump(all_testing_samples, f, indent=2)

print(f"\nSuccessfully generated {len(all_testing_samples)} testing samples.")
print(f"Data saved to: {OUTPUT_FILENAME}")