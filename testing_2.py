import json
from reward_engine import get_reward_engine

def test_reward_engine():
    """Test the reward engine with various scenarios."""
    print('Initializing test suite...')
    engine = get_reward_engine()
    
    # Test cases
    test_cases = [
        # 1. High engagement case - should get a reward
        {
            "test_id": 1,
            "description": "High engagement with streak and upvotes",
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
        # 2. Low activity - should not get a reward
        {
            "test_id": 2,
            "description": "Low activity user",
            "user_id": "stu_9002",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 1,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 2
            },
            "expected_result": False
        },
        # 3. Social butterfly - messaging and karma spending
        {
            "test_id": 3,
            "description": "Highly social user with karma spending",
            "user_id": "stu_9003",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 10,
                "upvotes_received": 5,
                "quizzes_completed": 0,
                "buddies_messaged": 8,
                "karma_spent": 40,
                "karma_earned_today": 30
            },
            "expected_result": True
        },
        # 4. Quiz master - completes multiple quizzes
        {
            "test_id": 4,
            "description": "User who completes many quizzes",
            "user_id": "stu_9004",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 7,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 3,
                "quizzes_completed": 4,
                "buddies_messaged": 0,
                "karma_spent": 10,
                "karma_earned_today": 40
            },
            "expected_result": True
        },
        # 5. Edge case - just above threshold
        {
            "test_id": 5,
            "description": "User just meeting reward thresholds",
            "user_id": "stu_9005",
            "date": "2024-07-20",
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
            "expected_result": True
        },
        # 6. Edge case - just below threshold
        {
            "test_id": 6,
            "description": "User just below reward thresholds",
            "user_id": "stu_9006",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 9,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 24,
                "karma_earned_today": 5
            },
            "expected_result": False
        },
        # 7. Return after long break
        {
            "test_id": 7,
            "description": "User returning after a long break",
            "user_id": "stu_9007",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 1,  # Just returned
                "posts_created": 0,
                "comments_written": 2,
                "upvotes_received": 3,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 10,
                "days_since_last_active": 10  # Custom field if supported
            },
            "expected_result": True
        },
        # 8. High karma earner
        {
            "test_id": 8,
            "description": "User earning lots of karma",
            "user_id": "stu_9008",
            "date": "2024-07-20",
            "daily_metrics": {
                "login_streak": 10,
                "posts_created": 3,
                "comments_written": 5,
                "upvotes_received": 25,
                "quizzes_completed": 2,
                "buddies_messaged": 3,
                "karma_spent": 50,
                "karma_earned_today": 100
            },
            "expected_result": True
        }
    ]

    # Run tests
    print("\nRunning test cases...\n")
    passed = 0
    failed = 0

    for test_case in test_cases:
        try:
            print(f"\nTest Case {test_case['test_id']}: {test_case['description']}")
            print("=" * 60)
            print("Input:")
            print(json.dumps(test_case["daily_metrics"], indent=2))
            
            # Get reward decision
            result = engine.check_surprise_box(
                test_case["user_id"],
                test_case["date"],
                test_case["daily_metrics"]
            )
            
            print("\nOutput:")
            print(json.dumps(result, indent=2))
            
            # Check if the result matches expected
            actual_result = result.get("surprise_unlocked", False)
            if actual_result == test_case["expected_result"]:
                status = "PASSED"
                passed += 1
            else:
                status = f"FAILED - Expected: {test_case['expected_result']}, Got: {actual_result}"
                failed += 1
            
            print(f"\nStatus: {status}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nError in test case {test_case.get('test_id', 'unknown')}: {str(e)}")
            failed += 1
            print("=" * 60)

    # Print summary
    print("\nTest Summary:")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(test_cases))*100:.2f}%")

if __name__ == "__main__":
    test_reward_engine()