# Dataset Creation Documentation

This document outlines the process for creating the train, validation, and test datasets for the Karma reward prediction model.

## Overview

The dataset is synthetically generated to simulate user behavior and reward distribution in a social platform. It creates realistic user activity patterns and applies reward conditions to generate labeled training data.

## Configuration

### User Clusters
Users are categorized into different behavioral clusters:
- **Casual**: Infrequent, low-activity users
- **Social**: Highly interactive users who engage with others
- **ContentCreator**: Users who frequently create posts and receive upvotes
- **QuizEnthusiast**: Users who actively participate in quizzes

Each cluster has different activity level parameters defined in `config.json`.

### Reward Conditions

Reward conditions are defined in `conditions.csv` with the following columns:
- `condition`: Logical expression based on user metrics
- `label`: Binary indicator (1=reward, 0=no reward)
- `reward_score`: Karma points awarded (if label=1)
- `reason`: Internal reason for reward
- `display_reason`: User-facing reward message
- `box_type`: Type of reward box
- `rarity`: Reward rarity (common, rare, legendary)
- `probability`: Likelihood of the condition being applied

## Data Generation Process

### 1. User Simulation
- Generate synthetic users with different behavioral patterns based on cluster distributions
- Each user has a 100-day activity history
- Daily metrics include:
  - `login_streak`: Consecutive days logged in
  - `posts_created`: Number of posts created
  - `comments_written`: Number of comments made
  - `upvotes_received`: Upvotes on user's content
  - `quizzes_completed`: Number of quizzes taken
  - `buddies_messaged`: Number of direct messages sent
  - `karma_spent`: Karma points spent
  - `karma_earned_today`: Karma points earned

### 2. Reward Assignment
For each user on each day:
1. Generate daily activity metrics based on user's cluster
2. Evaluate all reward conditions in `conditions.csv`
3. If multiple conditions are met, select one based on probability weights
4. Assign reward if condition is met (label=1)
5. Apply temporal trends (weekend/holiday effects)

### 3. Dataset Splitting
- **Training Set**: 70% of users (700 users)
- **Validation Set**: 15% of users (150 users)
- **Test Set**: 15% of users (150 users)

## Data Schema

Each sample in the dataset contains:
```json
{
  "user_id": "unique_user_identifier",
  "day": "YYYY-MM-DD",
  "features": {
    "login_streak": int,
    "posts_created": int,
    "comments_written": int,
    "upvotes_received": int,
    "quizzes_completed": int,
    "buddies_messaged": int,
    "karma_spent": int,
    "karma_earned_today": int
  },
  "label": 0|1,
  "reward_score": int,
  "box_type": string,
  "reason": string,
  "display_reason": string
}
```

## Usage

To generate new datasets:
1. Update `config.json` for different user behavior parameters
2. Modify `conditions.csv` to change reward conditions
3. Run `dataset.py` to generate new datasets

## Validation

The script includes validation to ensure:
- At least 10% of samples are positive (rewarded)
- No user has more than 30 consecutive days without rewards
- Reward distribution follows expected patterns

## Output Files
- `train_data.json`: Training dataset
- `val_data.json`: Validation dataset
- `test_data.json`: Test dataset

## Research Papers
1. Machine Learning for Synthetic Data Generation: A Review:https://arxiv.org/html/2302.04062v9

2. A Synthetic User Behavior Dataset Design for Data-driven AI-based Personalized Wireless Networks:https://www.researchgate.net/publication/331987705_A_Synthetic_User_Behavior_Dataset_Design_for_Data-driven_AI-based_Personalized_Wireless_Networks

3. Synthetic Data in AI: Challenges, Applications, and Ethical Implications:https://arxiv.org/html/2401.01629v1

4.User Modeling and User Profiling: A Comprehensive Survey:https://arxiv.org/html/2402.09660v2
