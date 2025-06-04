import json
from reward_engine import get_reward_engine

def test_reward_engine():
    """Test the reward engine with various scenarios."""
    print('daasa')
    engine = get_reward_engine()
    
    # Test cases
    test_cases = [
        # High engagement case - should get a reward
        {
            "user_id": "stu_9001",
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
            },
            "expected_result": True
        },
        # Low activity case - should not get a reward
        {
            "user_id": "stu_9002",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 3
            },
            "expected_result": False
        },
        # Social generosity case - should get a reward
        {
            "user_id": "stu_9003",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 2,
                "upvotes_received": 5,
                "quizzes_completed": 0,
                "buddies_messaged": 3,
                "karma_spent": 30,
                "karma_earned_today": 15
            },
            "expected_result": True
        },
        # Spammy behavior case - should not get a reward
        {
            "user_id": "stu_9004",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 6,
                "comments_written": 1,
                "upvotes_received": 1,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 5
            },
            "expected_result": False
        }
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"User ID: {test_case['user_id']}")
        print(f"Expected result: {'Reward' if test_case['expected_result'] else 'No Reward'}")
        
        result = engine.check_surprise_box(
            test_case["user_id"],
            test_case["date"],
            test_case["daily_metrics"]
        )
        
        print("Result:")
        print(json.dumps(result, indent=2))
        
        # Check if the result matches the expectation
        if result["surprise_unlocked"] == test_case["expected_result"]:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    test_reward_engine()