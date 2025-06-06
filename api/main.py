from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import sys
import os

# Add parent directory to path to import RewardEngine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward_engine import RewardEngine, get_reward_engine

# App configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class DailyMetrics(BaseModel):
    login_streak: int = 0
    posts_created: int = 0
    comments_written: int = 0
    upvotes_received: int = 0
    quizzes_completed: int = 0
    buddies_messaged: int = 0
    karma_spent: int = 0
    karma_earned_today: int = 0

class RewardRequest(BaseModel):
    user_id: str
    date: str  # Format: YYYY-MM-DD
    daily_metrics: DailyMetrics

class SurpriseBoxResponse(BaseModel):
    user_id: str
    surprise_unlocked: bool
    reward_karma: Optional[int] = None
    reason: Optional[str] = None
    rarity: Optional[str] = None
    box_type: Optional[str] = None
    status: str = "delivered"

class HealthResponse(BaseModel):
    status: str = "ok"

class VersionResponse(BaseModel):
    version: str
    model_version: str
    last_updated: str

# Simple user validation function for future use
def validate_user(user_id: str):
    """Placeholder for user validation logic"""
    return True

# Initialize FastAPI app
app = FastAPI(
    title="Karma Reward Engine API",
    description="API for Karma Reward Engine that determines surprise box rewards",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize reward engine
reward_engine = get_reward_engine()

# Authentication endpoint removed as per requirements

# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to Karma Reward Engine API"}

@app.post("/check-surprise-box", response_model=SurpriseBoxResponse)
async def check_surprise_box(
    request: RewardRequest
):
    """
    Check if a user qualifies for a surprise box and calculate the reward details.
    
    - **user_id**: The ID of the user
    - **date**: Date in YYYY-MM-DD format
    - **daily_metrics**: Dictionary containing the user's daily activity metrics
    
    Returns reward details including qualification status, karma points, and box details.
    """
    try:
        # Convert daily_metrics to dict for the reward engine
        metrics_dict = request.daily_metrics.dict()
        
        # Check for surprise box
        result = reward_engine.check_surprise_box(
            user_id=request.user_id,
            date=request.date,
            daily_metrics=metrics_dict
        )
        
        # The reward engine already formats the response correctly
        # Just ensure all required fields are present
        if not result.get("user_id"):
            result["user_id"] = request.user_id
        
        # Ensure all required fields are present
        result.setdefault("surprise_unlocked", False)
        result.setdefault("reason", "No activity matched")
        result.setdefault("rarity", "common")
        result.setdefault("box_type", "mystery")
        result.setdefault("status", "missed" if not result["surprise_unlocked"] else "delivered")
        
        # Remove any extra fields that might cause issues
        allowed_fields = [
            "user_id", "surprise_unlocked", "reward_karma", "reason",
            "rarity", "box_type", "status"
        ]
        return {k: v for k, v in result.items() if k in allowed_fields}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing surprise box check: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/version", response_model=VersionResponse)
async def version_info():
    """Version information endpoint"""
    return {
        "version": "1.0.0",
        "model_version": "1.0",
        "last_updated": "2024-06-06"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Example usage:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
