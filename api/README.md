Below is a highly detailed **README.md** file for your Karma Surprise Box microservice project. This README provides comprehensive information about the project, including the purpose, architecture, file descriptions, API details, setup instructions, and more. It is structured to be clear and professional, suitable for developers, stakeholders, or anyone interacting with the project.

---

# Karma Surprise Box Microservice

## Overview

The **Karma Surprise Box** is an AI-driven, offline microservice designed to enhance user engagement on the KarmaOS platform by rewarding users with surprise boxes based on their daily activities, karma behavior, and engagement patterns. The microservice evaluates user metrics using a combination of rule-based logic and a machine learning model (RandomForestClassifier) to determine eligibility for rewards, the type of reward box, associated karma points, and the rarity of the reward. Built with **FastAPI**, it provides a robust API backend, ensuring deterministic randomization, offline operation, and JSON-based input/output.

This project adheres to the requirements outlined in the provided specification, ensuring scalability, modularity, and ease of maintenance. The microservice is containerized for deployment using Docker and served via Uvicorn on port 8000.

---

## Table of Contents

1. [Project Objective](#project-objective)
2. [Features](#features)
3. [Architecture](#architecture)
4. [File Descriptions](#file-descriptions)
5. [API Endpoints](#api-endpoints)
6. [Configuration Details](#configuration-details)
7. [Reward Logic](#reward-logic)
8. [Setup and Installation](#setup-and-installation)
9. [Usage](#usage)
10. [Testing and Validation](#testing-and-validation)
11. [Constraints](#constraints)
12. [Deliverables](#deliverables)
13. [Timeline](#timeline)
14. [Dependencies](#dependencies)
15. [Future Improvements](#future-improvements)
16. [Contact](#contact)

---

## Project Objective

The Karma Surprise Box microservice aims to:
- Determine if a user qualifies for a surprise box on a given day based on their activity metrics.
- Assign a karma reward score, box type, and rarity level.
- Provide a human-readable reason for the reward to enhance user engagement.
- Make the platform addictive through randomized, yet deterministic, reinforcement driven by AI and user behavior.

The microservice operates offline, ensuring no external API calls, and uses deterministic randomization based on user ID and date to ensure reproducibility.

---

## Features

- **AI-Driven Reward System**: Combines rule-based logic with a RandomForestClassifier to evaluate user eligibility for rewards.
- **Deterministic Randomization**: Ensures consistent reward outcomes for the same user and date using a hashed seed.
- **Configurable Rules**: Reward rules and box types are defined in `config.json`, allowing easy modification without code changes.
- **Dynamic Karma Calculation**: Karma rewards are calculated based on box type, rarity, and user activity levels, constrained within configurable bounds.
- **FastAPI Backend**: Provides a robust, scalable API with automatic Swagger documentation (`/docs`).
- **Offline Operation**: No external dependencies or API calls, ensuring reliability and performance.
- **Docker Support**: Containerized for easy deployment and scalability.
- **Daily Reward Limits**: Ensures users receive only one reward per day to prevent abuse.

---

## Architecture

The microservice follows a modular architecture with the following components:
- **FastAPI Application** (`main.py`): Defines API endpoints, handles requests, and integrates with the reward engine.
- **Reward Engine** (`reward_engine.py`): Core logic for evaluating user metrics, determining rewards, and calculating karma points.
- **RandomForestClassifier** (`classifier_bal_1.pkl`): Trained machine learning model to predict reward eligibility based on user metrics.
- **Configuration** (`config.json`): Defines reward rules, box types, karma bounds, and other parameters.
- **Feature Names** (`feature_names.json`): Ensures consistent feature ordering for model input.
- **Conditions** (`conditions.csv`): Predefined conditions for rule-based reward eligibility.

The workflow is as follows:
1. The API receives a POST request to `/check-surprise-box` with user ID, date, and daily metrics.
2. The `RewardEngine` validates the input, checks for duplicate rewards, and prepares features for the model.
3. The RandomForestClassifier predicts the probability of awarding a reward.
4. If the probability exceeds the threshold, the engine evaluates rules to determine the box type, rarity, and karma reward.
5. The response is returned in JSON format with reward details.

---

## File Descriptions

Below is a detailed description of each file in the project:

### 1. `main.py`
- **Purpose**: The entry point for the FastAPI application, defining API endpoints and handling HTTP requests.
- **Key Components**:
  - **FastAPI Setup**: Configures the FastAPI app with CORS middleware and Swagger documentation.
  - **Endpoints**:
    - `GET /`: Welcome message.
    - `POST /check-surprise-box`: Processes reward requests and returns reward details.
    - `GET /health`: Returns system status.
    - `GET /version`: Returns version information.
    - (Commented out) Configuration management endpoints for updating rules and configurations.
  - **Pydantic Models**: Defines input/output schemas (e.g., `RewardRequest`, `SurpriseBoxResponse`) for validation.
  - **Dependencies**: Integrates with `RewardEngine` for reward processing.
- **Usage**: Run with `uvicorn main:app --host 0.0.0.0 --port 8000` to start the server.

### 2. `reward_engine.py`
- **Purpose**: Implements the core logic for evaluating user metrics, determining rewards, and calculating karma points.
- **Key Components**:
  - **RewardEngine Class**:
    - Initializes the RandomForestClassifier model and loads configurations.
    - Tracks rewarded users to prevent duplicates.
    - Maps box types to human-readable reasons.
  - **Methods**:
    - `_load_config`: Loads and validates `config.json`.
    - `_evaluate_rule`: Checks if user metrics satisfy reward rule conditions.
    - `_determine_box_type`: Selects the appropriate box type based on matched rules.
    - `_calculate_rarity`: Determines reward rarity using probability-weighted selection.
    - `_calculate_reward_karma`: Computes karma points based on box type, rarity, and activity.
    - `_prepare_features`: Prepares feature vectors for the model, including raw metrics, rule-based features, and temporal multipliers.
    - `check_surprise_box`: Main method to process reward requests and return results.
  - **Parsing Functions**: Includes functions to tokenize and evaluate conditions (e.g., `parse_condition`, `evaluate_expression`).
- **Dependencies**: Requires `config.json`, `feature_names.json`, `conditions.csv`, and `classifier_bal_1.pkl`.

### 3. `classifier_bal_1.pkl`
- **Purpose**: A serialized RandomForestClassifier model trained to predict reward eligibility.
- **Details**:
  - **Input Features**: User metrics (e.g., `login_streak`, `posts_created`), rule-based features, and temporal multipliers.
  - **Output**: Probability of awarding a reward (label: 1 = yes, 0 = no).
  - **Training**: Assumes training on anonymized or simulated data with balanced classes.
- **Usage**: Loaded by `reward_engine.py` to predict reward eligibility.

### 4. `config.json`
- **Purpose**: Defines the configuration for reward rules, box types, and system parameters.
- **Structure**:
  - `reward_probability_threshold`: Minimum probability (0.85) for awarding a reward.
  - `reward_rules`: Dictionary mapping box types (e.g., `streak_engager`) to conditions and descriptions.
  - `box_types`: Defines box types, their base karma, and rarity weights (e.g., `common`, `rare`, `elite`, `legendary`).
  - `karma_min`/`karma_max`: Bounds for karma rewards (10–50).
  - `metric_ranges`: Defines valid ranges for user metrics.
  - `user_clusters`: Statistical profiles for user types (e.g., `Casual`, `Social`) for potential analytics.
  - `cluster_probs`: Probabilities for user clusters (not used in current logic-utilized for model training).
  - `temporal_trends`: Seasonal multipliers for specific months to adjust reward probability.
- **Usage**: Loaded by `reward_engine.py` to guide reward logic

### 5. `feature_names.json`
- **Purpose**: Specifies the order of feature names for model input.
- **Content**: List of feature names (`login_streak`, `posts_created`, etc.).
- **Usage**: Ensures consistent feature ordering in `reward_engine.py` when preparing model input.
- **Scale**: Deliberately seperated feature names for flexibility to add new features later on.(Future -proofing)

### 6. `conditions.csv`
- **Purpose**: Contains predefined conditions for reward eligibility, used by the rule-based logic.
- **Structure**:
  - Columns: `condition`, `label`, `reward_score`, `reason`, `display_reason`, `box_type`, `rarity`, `probability`.
  - Example: `login_streak >= 3 and posts_created >= 1 and quizzes_completed >= 1` with a `common` rarity and `streak` box type.
- **Usage**: Loaded by `reward_engine.py` to evaluate rule-based features for the model.
- **Scale**: Deliberately seperated conditions for flexibility to add new conditions later on.(Future -proofing)
---

## API Endpoints

| Method | Endpoint                | Description                              | Input Example                                                                 | Output Example                                                                 |
|--------|-------------------------|------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| GET    | `/`                     | Returns a welcome message                | None                                                                          | `{"message": "Welcome to Karma Reward Engine API"}`                            |
| POST   | `/check-surprise-box`   | Checks if a user qualifies for a reward   | `{"user_id": "stu_9003", "date": "2024-07-20", "daily_metrics": {...}}`       | `{"user_id": "stu_9003", "surprise_unlocked": true, "reward_karma": 15, ...}` |
| GET    | `/health`               | Returns system health status             | None                                                                          | `{"status": "ok", "timestamp": "...", "service": "...", "version": "1.0.0"}`  |
| GET    | `/version`              | Returns version information              | None                                                                          | `{"version": "1.0.0", "model_version": "1.0", "last_updated": "..."}`         |

**Note**: Configuration management endpoints (`/config`, `/rules`, etc.) are commented out in `main.py` but can be enabled for dynamic configuration updates.

---

## Configuration Details

The `config.json` file is the central configuration hub, containing:
- **Reward Rules**: Conditions for each box type (e.g., `streak_engager` requires `login_streak >= 4`, `posts_created >= 2`, `quizzes_completed >= 2`).
- **Box Types**: Defines karma values and rarity weights for each box type (e.g., `mystery_enthusiast` has a base karma of 20 and high `legendary` weight).
- **Karma Bounds**: Ensures rewards stay between 10 and 50 karma points.
- **Metric Ranges**: Defines valid ranges for metrics to prevent invalid inputs.(ignores spammy or invalid metrics)
- **Temporal Trends**: Applies seasonal multipliers (e.g., 1.3x in December) and day-of-week multipliers (1.2x on weekends).

---

## Reward Logic

The reward logic combines rule-based evaluation and a RandomForestClassifier model:
1. **Feature Preparation**:
   - Raw metrics (e.g., `login_streak`, `posts_created`).
   - Rule-based features (from `conditions.csv`).
   - Temporal multiplier (based on day/week and month).
2. **Model Prediction**:
   - The classifier predicts the probability of awarding a reward.
   - If the probability exceeds `reward_probability_threshold` (0.85), a reward is considered.
3. **Box Type Determination**:
   - Evaluates rules in `config.json` to find matching conditions.
   - Selects the most specific rule (highest number of conditions) or uses deterministic randomization for ties.
4. **Rarity Calculation**:
   - Uses probability-weighted selection adjusted by prediction probability.
   - Rarity levels: `common`, `rare`, `elite`, `legendary`.
5. **Karma Calculation**:
   - Base karma from box type, multiplied by rarity multiplier (e.g., 2.0x for `legendary`) and activity bonus (0–50% based on metrics).
   - Constrained within `karma_min` and `karma_max`.
6. **Reason Assignment**:
   - Maps box types to predefined reasons (e.g., `streak_engager` → "Consistent logins + content and quiz activity").

---

## Setup and Installation

### Prerequisites
- **Python**: 3.8+
- **Docker**: For containerized deployment
- **Dependencies**: Listed in `requirements.txt` (assumed, not provided):
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `pandas`
  - `numpy`
  - `joblib`
  - `python-multipart` (for potential file uploads)

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd karma-surprise-box
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure Required Files**:
   - Place `config.json`, `feature_names.json`, `conditions.csv`, and `classifier_bal_1.pkl` in the project root.
4. **Run the Application**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
5. **Docker Setup**:
   - Create a `Dockerfile` (example below):
     ```dockerfile
     FROM python:3.8
     WORKDIR /app
     COPY . .
     RUN pip install -r requirements.txt
     EXPOSE 8000
     CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
     ```
   - Build and run:
     ```bash
     docker build -t karma-surprise-box .
     docker run -p 8000:8000 karma-surprise-box
     ```

---

## Usage

### API Call Example
**Request**:
```bash
curl -X POST http://localhost:8000/check-surprise-box \
-H "Content-Type: application/json" \
-d '{
    "user_id": "stu_9003",
    "date": "2024-07-20",
    "daily_metrics": {
        "login_streak": 5,
        "posts_created": 2,
        "comments_written": 4,
        "upvotes_received": 11,
        "quizzes_completed": 1,
        "buddies_messaged": 2,
        "karma_spent": 35,
        "karma_earned_today": 50
    }
}'
```

**Response**:
```json
{
    "user_id": "stu_9003",
    "surprise_unlocked": true,
    "reward_karma": 15,
    "reason": "Engagement streak + high upvotes",
    "rarity": "rare",
    "box_type": "mystery",
    "status": "delivered"
}
```

### Accessing Swagger Docs
- Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Testing and Validation

The microservice was validated against the following test cases:
| Test Case                            | Expected Outcome                     |
|--------------------------------------|--------------------------------------|
| High engagement + streak             | Surprise box awarded                 |
| Low activity                         | No box awarded                      |
| Same input repeatedly                | Only one reward per day per user     |
| Random rewards                       | Controlled via deterministic seeds   |

**Validation Steps**:
1. Test `/check-surprise-box` with various metric combinations.
2. Verify duplicate reward prevention using the same `user_id` and `date`.
3. Check rarity distribution aligns with `config.json` weights.
4. Ensure karma rewards stay within `karma_min` and `karma_max`.

---

## Constraints

- **Offline Operation**: No external API calls are made.
- **Deterministic Randomization**: Uses MD5 hash of `user_id` and `date` for reproducibility.
- **JSON I/O**: All inputs and outputs are JSON-based.
- **No UI**: Pure API backend, with Swagger UI for documentation.
- **Single Reward per Day**: Enforced by tracking rewarded users in memory.

---

## Deliverables

- **Source Code**:
  - `main.py`
  - `reward_engine.py`
  - `classifier_bal_1.pkl`
- **Configuration Files**:
  - `config.json`
  - `feature_names.json`
  - `conditions.csv`
- **Documentation**:
  - This README.md
  - API call samples (above)
  - Docker run instructions
  - Reward logic explanation
- **Training Samples**: Assumed to be included in the training process for `classifier_bal_1.pkl` (not provided).

---

## Timeline

The project was developed over 6 weeks:
| Week | Tasks                                      |
|------|--------------------------------------------|
| 1    | Simulate training data, train classifier   |
| 2    | Build scoring and reward logic            |
| 3    | Develop FastAPI service                   |
| 4    | Add configuration and rarity logic         |
| 5    | Test and validate                         |
| 6    | Finalize Docker setup and documentation    |

---

## Dependencies

- **Python Libraries**:
  - `fastapi`: API framework
  - `uvicorn`: ASGI server
  - `pydantic`: Data validation
  - `pandas`: Data manipulation
  - `numpy`: Numerical operations
  - `joblib`: Model loading
- **Docker**: For containerization

---

## Future Improvements

- **Dynamic Configuration Updates**: Enable commented-out endpoints (`/config`, `/rules`) for real-time configuration changes.
- **Enhanced Model Training**: Incorporate more diverse training data to improve classifier accuracy.
- **Analytics Dashboard**: Add an endpoint to retrieve reward statistics for admin users.
- **Persistent Storage**: Store rewarded user data in a database instead of memory for scalability.
- **Advanced Temporal Features**: Incorporate more granular temporal trends (e.g., time of day).

---

## Contact

For questions or support, contact the development team at [your-email@example.com] or submit an issue on the project repository.

---

This README provides a comprehensive overview of the Karma Surprise Box microservice, ensuring that developers and users can understand, deploy, and extend the system effectively. Let me know if you need adjustments or additional details!