# Karma Surprise Box API

A FastAPI-based microservice that determines if a user qualifies for a Karma Surprise Box based on their daily activity, karma behavior, and engagement patterns.

## Features

- Secure authentication with JWT tokens
- RESTful endpoints for checking rewards
- Input validation using Pydantic models
- CORS support
- Health check endpoint

## Setup

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the `api` directory with:
   ```
   SECRET_KEY=your-secret-key-here
   ```

## Running the API

Start the FastAPI development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## Authentication

1. Get an access token:
   ```
   POST /token
   Form Data: username=testuser&password=secret
   ```

2. Use the token in subsequent requests:
   ```
   Authorization: Bearer <your-token>
   ```

## API Endpoints

### Check Surprise Box

```
POST /check-surprise-box
```

**Request Body:**
```json
{
  "user_id": "user123",
  "date": "2023-06-06",
  "daily_metrics": {
    "login_streak": 5,
    "posts_created": 2,
    "comments_written": 3,
    "upvotes_received": 10,
    "quizzes_completed": 1,
    "buddies_messaged": 2,
    "karma_spent": 5,
    "karma_earned_today": 15
  }
}
```

**Success Response:**
```json
{
  "user_id": "user123",
  "surprise_unlocked": true,
  "reward_karma": 15,
  "reason": "Engagement reward",
  "rarity": "rare",
  "box_type": "mystery",
  "status": "delivered"
}
```

**No Reward Response:**
```json
{
  "user_id": "user123",
  "surprise_unlocked": false,
  "status": "missed"
}
```

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

### Version Information

```
GET /version
```

**Response:**
```json
{
  "version": "1.0.0",
  "model_version": "1.0",
  "last_updated": "2024-06-06"
}
```

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-06-06T12:00:00.000000"
}
```

## Testing

You can test the API using the interactive documentation at `http://localhost:8000/docs` or using a tool like `curl` or Postman.

## Deployment

For production deployment, consider using:
- Gunicorn with Uvicorn workers
- Nginx as a reverse proxy
- Environment variables for configuration
- Proper SSL/TLS setup

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| SECRET_KEY | Secret key for JWT token signing | (required) |
| ACCESS_TOKEN_EXPIRE_MINUTES | Token expiration time in minutes | 30 |

## License

[Your License Here]
