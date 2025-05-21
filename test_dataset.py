import csv
import re
import random
import json

# Set random seed for reproducibility
random.seed(42)

# Define reasonable ranges for each user metric
metric_ranges = {
    "login_streak": (0, 30),        # Days
    "posts_created": (0, 20),       # Number of posts
    "comments_written": (0, 50),    # Number of comments
    "upvotes_received": (0, 100),   # Number of upvotes
    "quizzes_completed": (0, 10),   # Number of quizzes
    "buddies_messaged": (0, 20),    # Number of messages
    "karma_spent": (0, 100),        # Karma points spent
    "karma_earned_today": (0, 200)  # Karma points earned
}

def parse_condition(condition_str):
    """
    Parse a condition string into a dictionary of constraints.
    Example: "login_streak >= 7 and posts_created >= 1" -> 
             {"login_streak": (">=", 7), "posts_created": (">=", 1)}
    """
    constraints = {}
    parts = condition_str.split(" and ")
    for part in parts:
        match = re.match(r"(\w+)\s*(>=|<=|==)\s*(\d+)", part.strip())
        if match:
            metric, operator, value = match.groups()
            constraints[metric] = (operator, int(value))
        else:
            raise ValueError(f"Invalid constraint: {part}")
    return constraints

def generate_features(constraints, metric_ranges):
    """
    Generate a feature dictionary satisfying the given constraints.
    Metrics without constraints get random values from their full range.
    """
    features = {}
    for metric, (min_val, max_val) in metric_ranges.items():
        if metric in constraints:
            operator, value = constraints[metric]
            if operator == "==":
                features[metric] = value
            elif operator == ">=":
                features[metric] = random.randint(value, max_val)
            elif operator == "<=":
                features[metric] = random.randint(min_val, value)
        else:
            features[metric] = random.randint(min_val, max_val)
    return features

# Number of samples to generate per condition for testing
num_samples_per_condition = 20

# Initialize the testing dataset
testing_dataset = []

# Read the CSV file (assumed to be named "conditions.csv")
with open("conditions.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        condition_str = row["condition"]
        constraints = parse_condition(condition_str)
        # Generate data points for testing
        for _ in range(num_samples_per_condition):
            features = generate_features(constraints, metric_ranges)
            data_point = {
                "features": features,
                "label": int(row["label"]),
                "reward_score": int(row["reward_score"]),
                "box_type": row["box_type"],
                "rarity": row["rarity"]
            }
            testing_dataset.append(data_point)

# Save the testing dataset to a JSON file
with open("testing_data.json", "w") as jsonfile:
    json.dump(testing_dataset, jsonfile, indent=2)

print("Testing dataset generated and saved to 'testing_data.json'.")