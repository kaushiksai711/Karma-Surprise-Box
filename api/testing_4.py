import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from reward_engine import get_reward_engine, RewardEngine
import hashlib
import pytest

# Set random seed for reproducibility
random.seed(42)

def generate_test_cases() -> list[Dict[str, Any]]:
    """Generate comprehensive test cases for the reward engine."""
    test_cases = []

    def get_date(days_ago=0) -> str:
        return (datetime.now()).strftime("%Y-%m-%d")

    def get_deterministic_seed(user_id: str, date: str) -> int:
        """Generate a deterministic seed based on user_id and date."""
        seed_str = f"{user_id}_{date}"
        hash_value = hashlib.md5(seed_str.encode()).hexdigest()
        return int(hash_value[:8], 16)

    # 1-10: Basic engagement patterns (covering high engagement, low activity)
    test_cases.extend([
        # 1. High engagement across all metrics
        {
            "test_id": 1,
            "description": "High engagement across all metrics",
            "user_id": "user_1001",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 7,
                "posts_created": 3,
                "comments_written": 5,
                "upvotes_received": 15,
                "quizzes_completed": 2,
                "buddies_messaged": 4,
                "karma_spent": 30,
                "karma_earned_today": 60
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["streak_engager", "balanced_contributor", "community_champion"],
                "status": "delivered"
            }
        },
        # 2. Minimal activity - below all thresholds
        {
            "test_id": 2,
            "description": "Minimal activity - below all thresholds",
            "user_id": "user_1002",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 1,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 3. Social butterfly - high messaging and comments
        {
            "test_id": 3,
            "description": "Highly social user with many messages and comments",
            "user_id": "user_1003",
            "date": get_date(1),
            "daily_metrics": {
                "login_streak": 5,
                "posts_created": 1,
                "comments_written": 10,
                "upvotes_received": 8,
                "quizzes_completed": 0,
                "buddies_messaged": 8,
                "karma_spent": 20,
                "karma_earned_today": 30
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["community_glue"],
                "status": "delivered"
            }
        },
        # 4. Quiz enthusiast - completes many quizzes
        {
            "test_id": 4,
            "description": "User completing multiple quizzes",
            "user_id": "user_1004",
            "date": get_date(2),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 5,
                "quizzes_completed": 4,
                "buddies_messaged": 0,
                "karma_spent": 5,
                "karma_earned_today": 40
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["quiz_enthusiast", "knowledge_contributor"],
                "status": "delivered"
            }
        },
        # 5. Content creator - many posts and upvotes
        {
            "test_id": 5,
            "description": "Content creator with many posts and upvotes",
            "user_id": "user_1005",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 10,
                "posts_created": 4,
                "comments_written": 2,
                "upvotes_received": 25,
                "quizzes_completed": 1,
                "buddies_messaged": 1,
                "karma_spent": 10,
                "karma_earned_today": 50
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["streak_engager"],
                "status": "delivered"
            }
        },
        # 6. Comeback user
        {
            "test_id": 6,
            "description": "User returning after a break",
            "user_id": "user_1006",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 1,
                "comments_written": 2,
                "upvotes_received": 8,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 15
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["creative_scholar","rising_star"],
                "status": "delivered"
            }
        },
        # 7. Karma trader - high karma spent and earned
        {
            "test_id": 7,
            "description": "User with balanced karma activity",
            "user_id": "user_1007",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 5,
                "quizzes_completed": 0,
                "buddies_messaged": 2,
                "karma_spent": 20,
                "karma_earned_today": 25
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["active_supporter"],
                "status": "delivered"
            }
        },
        # 8. Balanced contributor
        {
            "test_id": 8,
            "description": "Well-rounded contributor with balanced metrics",
            "user_id": "user_1008",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 4,
                "posts_created": 2,
                "comments_written": 3,
                "upvotes_received": 12,
                "quizzes_completed": 1,
                "buddies_messaged": 2,
                "karma_spent": 15,
                "karma_earned_today": 25
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["balanced_contributor", "creative_scholar"],
                "status": "delivered"
            }
        },
        # 9. Minimal qualifying activity
        {
            "test_id": 9,
            "description": "Barely meets minimum requirements",
            "user_id": "user_1009",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 1,
                "comments_written": 1,
                "upvotes_received": 10,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 8
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["mystery"],
                "status": "delivered"
            }
        },
        # 10. Just below thresholds
        {
            "test_id": 10,
            "description": "Just below all thresholds",
            "user_id": "user_1010",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 9,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 24,
                "karma_earned_today": 4
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        }
    ])

    # 11-20: Edge cases and specific conditions
    test_cases.extend([
        # 11. Exactly at login streak threshold
        {
            "test_id": 11,
            "description": "Login streak exactly at threshold (3 days)",
            "user_id": "user_10110",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 0,
                "upvotes_received": 10,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 15
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["creative_scholar","rising_star"],
                "status": "delivered"
            }
        },
        # 12. Just below login streak threshold
        {
            "test_id": 12,
            "description": "Login streak just below threshold",
            "user_id": "user_1012",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 1,
                "comments_written": 1,
                "upvotes_received": 10,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 15
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["rising_star"],
                "status": "delivered"
            }
        },
        # 13. High upvotes with minimal other activity
        {
            "test_id": 13,
            "description": "High upvotes with minimal other activity",
            "user_id": "user_1013",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 15,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 5
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 14. Many comments but no upvotes
        {
            "test_id": 14,
            "description": "Many comments but no upvotes",
            "user_id": "user_1014",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 0,
                "comments_written": 8,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 5
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 15. High karma spent but no other activity
        {
            "test_id": 15,
            "description": "High karma spent but no other activity",
            "user_id": "user_1015",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 30,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 16. Perfectly balanced engagement
        {
            "test_id": 16,
            "description": "Perfectly balanced engagement",
            "user_id": "user_1016",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 4,
                "posts_created": 1,
                "comments_written": 2,
                "upvotes_received": 8,
                "quizzes_completed": 1,
                "buddies_messaged": 2,
                "karma_spent": 15,
                "karma_earned_today": 20
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["streak_engager"],
                "status": "delivered"
            }
        },
        # 17. Long streak but minimal activity
        {
            "test_id": 17,
            "description": "Long streak with minimal activity",
            "user_id": "user_1017",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 30,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 1,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 1
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 18. High karma earned but no other activity
        {
            "test_id": 18,
            "description": "High karma earned but no other activity",
            "user_id": "user_1018",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 50
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 19. High activity but no karma spent
        {
            "test_id": 19,
            "description": "High activity but no karma spent",
            "user_id": "user_1019",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 5,
                "posts_created": 2,
                "comments_written": 3,
                "upvotes_received": 12,
                "quizzes_completed": 1,
                "buddies_messaged": 2,
                "karma_spent": 0,
                "karma_earned_today": 25
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["balanced_contributor", "creative_scholar"],
                "status": "delivered"
            }
        },
        # 20. High karma spent with minimal activity
        {
            "test_id": 20,
            "description": "High karma spent with minimal activity",
            "user_id": "user_1020",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 2,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 40,
                "karma_earned_today": 5
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        }
    ])

    # 21-30: Specific reward conditions (from conditions.csv)
    test_cases.extend([
        # 21. Streak engager (login_streak >= 3 and posts_created >= 1 and quizzes_completed >= 1)
        {
            "test_id": 21,
            "description": "Streak engager condition match",
            "user_id": "user_1021",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["streak_engager"],
                "status": "delivered"
            }
        },
        # 22. Quiz enthusiast (quizzes_completed >= 2 and login_streak >= 2)
        {
            "test_id": 22,
            "description": "Quiz enthusiast condition match",
            "user_id": "user_1022",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 4,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 5,
                "buddies_messaged": 0,
                "karma_spent": 13,
                "karma_earned_today": 13
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["quiz_completion"],
                "status": "delivered"
            }
        },
        # 23. Community champion (posts_created >= 2 and upvotes_received >= 12 and buddies_messaged >= 2)
        {
            "test_id": 23,
            "description": "Community champion condition match",
            "user_id": "user_1023",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 2,
                "comments_written": 0,
                "upvotes_received": 12,
                "quizzes_completed": 0,
                "buddies_messaged": 2,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["community_champion"],
                "status": "delivered"
            }
        },
        # 24. No reward - spammy behavior (posts_created >= 5 and upvotes_received <= 2)
        {
            "test_id": 24,
            "description": "Spammy behavior - no reward",
            "user_id": "user_1024",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 5,
                "comments_written": 0,
                "upvotes_received": 2,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 25. No reward - very low activity
        {
            "test_id": 25,
            "description": "Very low activity - no reward",
            "user_id": "user_1025",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 26. Social influencer (buddies_messaged >= 2 and comments_written >= 2 and upvotes_received >= 5)
        {
            "test_id": 26,
            "description": "Social influencer condition match",
            "user_id": "user_1026",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 0,
                "comments_written": 2,
                "upvotes_received": 5,
                "quizzes_completed": 0,
                "buddies_messaged": 2,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["mystery","social_butterfly","quiz_completion"],
                "status": "delivered"
            }
        },
        # 27. Creative scholar (posts_created >= 1 and quizzes_completed >= 1 and upvotes_received >= 3)
        {
            "test_id": 27,
            "description": "Creative scholar condition match",
            "user_id": "user_1027",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 1,
                "comments_written": 0,
                "upvotes_received": 3,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["creative_scholar"],
                "status": "delivered"
            }
        },
        # 28. Community glue (buddies_messaged >= 4 and karma_spent >= 8 and login_streak >= 2)
        {
            "test_id": 28,
            "description": "Community glue condition match",
            "user_id": "user_1028",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 4,
                "karma_spent": 8,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["community_glue"],
                "status": "delivered"
            }
        },
        # 29. Active supporter (login_streak >= 3 and karma_earned_today >= 8)
        {
            "test_id": 29,
            "description": "Active supporter condition match",
            "user_id": "user_1029",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 8
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["active_supporter"],
                "status": "delivered"
            }
        },
        # 30. Mystery enthusiast (quizzes_completed >= 1 and (posts_created >= 1 or comments_written >= 2))
        {
            "test_id": 30,
            "description": "Mystery enthusiast condition match",
            "user_id": "user_1030",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 2,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 2,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["mystery_enthusiast","quiz_enthusiast","quiz_completion"],
                "status": "delivered"
            }
        }
    ])

    # 31-40: Boundary value testing
    test_cases.extend([
        # 31. Maximum values for all metrics
        {
            "test_id": 31,
            "description": "Upper boundary values for all metrics",
            "user_id": "user_1031",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 30,
                "posts_created": 10,
                "comments_written": 20,
                "upvotes_received": 100,
                "quizzes_completed": 5,
                "buddies_messaged": 30,
                "karma_spent": 50,
                "karma_earned_today": 100
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["community_champion", "balanced_contributor"],
                "status": "delivered"
            }
        },
        # 32. Minimum values for all metrics
        {
            "test_id": 32,
            "description": "Minimum values for all metrics",
            "user_id": "user_1032",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 33. Just above karma_spent threshold
        {
            "test_id": 33,
            "description": "Just above karma_spent threshold",
            "user_id": "user_1033",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 34. Just below karma_spent threshold
        {
            "test_id": 34,
            "description": "Just below karma_spent threshold",
            "user_id": "user_1034",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 24,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 35. Maximum login streak
        {
            "test_id": 35,
            "description": "Maximum login streak",
            "user_id": "user_1035",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 30,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 36. Maximum upvotes received
        {
            "test_id": 36,
            "description": "Maximum upvotes received",
            "user_id": "user_1036",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 6,
                "comments_written": 3,
                "upvotes_received": 100,
                "quizzes_completed": 0,
                "buddies_messaged": 13,
                "karma_spent": 25,
                "karma_earned_today": 50
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["creative_scholar","mystery_enthusiast","mystery"],
                "status": "delivered"
            }
        },
        # 37. Just above quizzes completed threshold
        {
            "test_id": 37,
            "description": "Just above quizzes completed threshold",
            "user_id": "user_1037",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 1,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 5,
                "buddies_messaged": 1,
                "karma_spent": 20,
                "karma_earned_today":25
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["quiz_enthusiast"],
                "status": "delivered"
            }
        },
        # 38. Just below quizzes completed threshold
        {
            "test_id": 38,
            "description": "Just below quizzes completed threshold",
            "user_id": "user_1038",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 39. Maximum buddies messaged
        {
            "test_id": 39,
            "description": "Maximum buddies messaged",
            "user_id": "user_1039",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 30,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 40. Just above comments written threshold
        {
            "test_id": 40,
            "description": "Just above comments written threshold",
            "user_id": "user_1040",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 8,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        }
    ])

    # 41-50: Realistic scenarios and special cases
    test_cases.extend([
        # 41. New user with strong first day
        {
            "test_id": 41,
            "description": "New user with strong first day engagement",
            "user_id": "user_1041",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 1,
                "comments_written": 3,
                "upvotes_received": 8,
                "quizzes_completed": 1,
                "buddies_messaged": 1,
                "karma_spent": 10,
                "karma_earned_today": 15
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["creative_scholar"],
                "status": "delivered"
            }
        },
        # 42. Rising star
        {
            "test_id": 42,
            "description": "Rising star with high upvotes and karma",
            "user_id": "user_1042",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 6,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 12
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["rising_star"],
                "status": "delivered"
            }
        },
        # # 43. Deterministic randomization test
        # {
        #     "test_id": 43,
        #     "description": "Verify deterministic randomization for same user_id and date",
        #     "user_id": "user_1043",
        #     "date": get_date(0),
        #     "daily_metrics": {
        #         "login_streak": 4,
        #         "posts_created": 2,
        #         "comments_written": 3,
        #         "upvotes_received": 12,
        #         "quizzes_completed": 1,
        #         "buddies_messaged": 2,
        #         "karma_spent": 15,
        #         "karma_earned_today": 25
        #     },
        #     "expected_result": {
        #         "surprise_unlocked": True,
        #         "reward_karma": lambda x: 10 <= x <= 50,
        #         "rarity": ["common", "rare", "elite", "legendary"],
        #         "box_type": ["balanced_contributor", "creative_scholar"],
        #         "status": "delivered"
        #     },
        #     "repeat_test": True  # Indicates this test should be run twice to check consistency
        # },
        #44. Same input repeatedly test
        {
            "test_id": 44,
            "description": "Same input repeatedly for same user_id and date",
            "user_id": "user_1044",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 1,
                "upvotes_received": 10,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 15
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["rising_star","creative_scholar"],
                "status": "delivered"
            },
            "repeat_test": True  # Run twice to simulate repeated calls
        },
        # 45. Temporal multiplier effect (weekend)
        {
            "test_id": 45,
            "description": "Test temporal multiplier on a weekend",
            "user_id": "user_1045",
            "date": "2025-06-21",  # Saturday
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 1,
                "upvotes_received": 10,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 15
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["rising_star"],
                "status": "delivered"
            }
        },
        # 46. Temporal multiplier effect (seasonal - December)
        {
            "test_id": 46,
            "description": "Test temporal multiplier in December",
            "user_id": "user_1046",
            "date": "2025-12-01",
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 1,
                "upvotes_received": 10,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 15
            },
            "expected_result": {
                "surprise_unlocked": True,
                "reward_karma": lambda x: 10 <= x <= 50,
                "rarity": ["common", "rare", "elite", "legendary"],
                "box_type": ["creative_scholar"],
                "status": "delivered"
            }
        },
        # 47. Invalid date format
        {
            "test_id": 47,
            "description": "Invalid date format",
            "user_id": "user_1047",
            "date": "2025-13-01",  # Invalid month
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 1,
                "upvotes_received": 10,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 15
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Error processing request: time data '2025-13-01' does not match format '%Y-%m-%d'",
                "rarity": None,
                "box_type": None,
                "status": "error"
            }
        },
        # 48. Missing metrics
        # {
        #     "test_id": 48,
        #     "description": "Missing some metrics",
        #     "user_id": "user_1048",
        #     "date": get_date(0),
        #     "daily_metrics": {
        #         "login_streak": 3,
        #         "posts_created": 1,
        #         "quizzes_completed": 1
        #         # Missing other metrics, should default to 0
        #     },
        #     "expected_result": {
        #         "surprise_unlocked": True,
        #         "reward_karma": lambda x: 10 <= x <= 50,
        #         "rarity": ["common", "rare", "elite", "legendary"],
        #         "box_type": ["creative_scholar"],
        #         "status": "delivered"
        #     }
        # },
        # 49. Negative metric values
        {
            "test_id": 49,
            "description": "Negative metric values",
            "user_id": "user_1049",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": -1,
                "posts_created": -1,
                "comments_written": -1,
                "upvotes_received": -1,
                "quizzes_completed": -1,
                "buddies_messaged": -1,
                "karma_spent": -1,
                "karma_earned_today": -1
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        },
        # 50. Reward probability threshold edge case
        {
            "test_id": 50,
            "description": "Edge case near reward probability threshold",
            "user_id": "user_1050",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": {
                "surprise_unlocked": False,
                "reward_karma": None,
                "reason": "Activity level below reward threshold",
                "rarity": None,
                "box_type": None,
                "status": "missed"
            }
        }
    ])

    return test_cases

def validate_result(result: Dict[str, Any], expected: Dict[str, Any], test_id: int) -> tuple[bool, str]:
    """
    Validate the result against the expected output.
    
    Args:
        result: Actual result from reward engine
        expected: Expected result from test case
        test_id: Test ID for error reporting
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []

    # Check user_id
    if result.get("user_id") != expected.get("user_id", result.get("user_id")):
        errors.append(f"Test {test_id}: Incorrect user_id. Expected {expected.get('user_id')}, got {result.get('user_id')}")

    # Check surprise_unlocked
    if result.get("surprise_unlocked") != expected.get("surprise_unlocked"):
        errors.append(f"Test {test_id}: Incorrect surprise_unlocked. Expected {expected['surprise_unlocked']}, got {result['surprise_unlocked']}")

    # Check reward_karma
    if expected["reward_karma"] is not None:
        if callable(expected["reward_karma"]):
            if not expected["reward_karma"](result.get("reward_karma", 0)):
                errors.append(f"Test {test_id}: reward_karma out of range. Got {result.get('reward_karma')}")
        elif result.get("reward_karma") != expected["reward_karma"]:
            errors.append(f"Test {test_id}: Incorrect reward_karma. Expected {expected['reward_karma']}, got {result.get('reward_karma')}")
    elif result.get("reward_karma") is not None:
        errors.append(f"Test {test_id}: reward_karma should be None, got {result.get('reward_karma')}")

    # Check reason
    # if expected["reason"] is not None:
    #     if callable(expected["reason"]):
    #         if not expected["reason"](result.get("reason", "")):
    #             errors.append(f"Test {test_id}: reason does not match expected pattern. Got {result.get('reason')}")
    #     elif result.get("reason") != expected["reason"]:
    #         errors.append(f"Test {test_id}: Incorrect reason. Expected {expected['reason']}, got {result.get('reason')}")

    # Check rarity
    if expected["rarity"] is not None:
        if result.get("rarity") not in expected["rarity"]:
            errors.append(f"Test {test_id}: Incorrect rarity. Expected one of {expected['rarity']}, got {result.get('rarity')}")
    elif result.get("rarity") is not None:
        errors.append(f"Test {test_id}: rarity should be None, got {result.get('rarity')}")

    # Check status
    if result.get("status") != expected["status"]:
        errors.append(f"Test {test_id}: Incorrect status. Expected {expected['status']}, got {result.get('status')}")

    return len(errors) == 0, "; ".join(errors) if errors else ""

@pytest.fixture(scope="module")
def reward_engine() -> RewardEngine:
    """Fixture to provide a single reward engine instance for all tests."""
    return get_reward_engine()

def test_reward_engine(reward_engine: RewardEngine):
    """Execute all test cases and report results."""
    print("🏆 Starting Reward Engine Test Suite")
    print("=" * 50)

    test_cases = generate_test_cases()
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "test_details": []
    }

    for test in test_cases:
        try:
            print(f"\n🏆 Test {test['test_id']}: {test['description']}")
            print("-" * 60)

            # Execute test
            result = reward_engine.check_surprise_box(
                test["user_id"],
                test["date"],
                test["daily_metrics"]
            )

            # Validate result
            test_passed, error_msg = validate_result(result, test["expected_result"], test["test_id"])

            # Handle repeat test for deterministic randomization or same input
            if test.get("repeat_test"):
                # Run the test again with the same input
                result_repeat = reward_engine.check_surprise_box(
                    test["user_id"],
                    test["date"],
                    test["daily_metrics"]
                )
                # For deterministic randomization (Test 43), check if results are identical
                if test["test_id"] == 43:
                    if result != result_repeat:
                        test_passed = False
                        error_msg += f"; Test {test['test_id']}: Non-deterministic output for same input"
                # For same input repeatedly (Test 44), check if second call behaves appropriately
                elif test["test_id"] == 44:
                    # Assuming the engine prevents duplicate rewards (implementation-dependent)
                    if result_repeat["surprise_unlocked"]:
                        test_passed = False
                        error_msg += f"; Test {test['test_id']}: Second call should not award a reward"

            # Update results
            if test_passed:
                results["passed"] += 1
                status = "✅ PASSED"
            else:
                results["failed"] += 1
                status = "❌ FAILED"

            # Store test details
            test_result = {
                "test_id": test["test_id"],
                "description": test["description"],
                "status": "PASSED" if test_passed else "FAILED",
                "expected": test["expected_result"],
                "actual": result,
                "error": error_msg,
                "reward_details": result
            }
            results["test_details"].append(test_result)

            # Print test result
            print(f"{status}")
            if error_msg:
                print(f"Errors: {error_msg}")
            print(f"Result: {result}")

        except Exception as e:
            print(f"❌ ERROR in test {test.get('test_id', 'unknown')}: {str(e)}")
            results["failed"] += 1
            results["test_details"].append({
                "test_id": test.get("test_id", "unknown"),
                "status": "ERROR",
                "error": str(e)
            })

    # Print summary
    print("\n" + "=" * 50)
    print("🏆 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {results['total']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    pass_rate = (results['passed'] / results['total']) * 100 if results['total'] > 0 else 0
    print(f"🏆 Pass Rate: {pass_rate:.2f}%")

    # Save detailed results
    def default_serializer(obj):
        if callable(obj):
            return f"<function {obj.__name__ if hasattr(obj, '__name__') else 'lambda'}>"
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        
    with open('test_results_detailed.json', 'w') as f:
        json.dump(results, f, default=default_serializer, indent=2)
    print("\n🏆 Detailed results saved to 'test_results_detailed.json'")

    # Assert that all tests passed
    assert results["failed"] == 0, f"{results['failed']} tests failed"

if __name__ == "__main__":
    pytest.main(["-v", __file__])