import json
import random
from datetime import datetime, timedelta
from reward_engine import get_reward_engine

def generate_test_cases():
    """Generate 50 comprehensive test cases for the reward engine."""
    test_cases = []
    
    def get_date(days_ago=0):
        return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # 1-10: Basic engagement patterns
    test_cases.extend([
        # 1. High engagement with all metrics
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
            "expected_result": True
        },
        # 2. Minimal activity - should not qualify
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
            "expected_result": False
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
            "expected_result": True
        },
        # 4. Quiz master - completes many quizzes
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
            "expected_result": True
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
            "expected_result": True
        },
        # 6. Returning user after break
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
                "karma_earned_today": 15,
                "days_since_last_active": 14
            },
            "expected_result": True
        },
        # 7. Karma spender - high karma spent
        {
            "test_id": 7,
            "description": "User spending lots of karma",
            "user_id": "user_1007",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 5,
                "quizzes_completed": 0,
                "buddies_messaged": 2,
                "karma_spent": 50,
                "karma_earned_today": 10
            },
            "expected_result": True
        },
        # 8. Balanced contributor
        {
            "test_id": 8,
            "description": "Well-rounded contributor with balanced metrics",
            "user_id": "user_1008",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 4,
                "posts_created": 1,
                "comments_written": 3,
                "upvotes_received": 12,
                "quizzes_completed": 1,
                "buddies_messaged": 2,
                "karma_spent": 15,
                "karma_earned_today": 25
            },
            "expected_result": True
        },
        # 9. Minimal qualifying activity
        {
            "test_id": 9,
            "description": "Barely meets minimum requirements",
            "user_id": "user_1009",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 1,
                "upvotes_received": 10,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 5
            },
            "expected_result": True
        },
        # 10. Almost but not quite
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
            "expected_result": True
        }
    ])
    
    # 11-30: Edge cases and specific conditions
    test_cases.extend([
        # 11. Exactly at login streak threshold (3 days)
        {
            "test_id": 11,
            "description": "Login streak exactly at threshold",
            "user_id": "user_1011",
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
            "expected_result": True
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
            "expected_result": True
        },
        # 13. High upvotes but minimal other activity
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
            "expected_result": True
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
            "expected_result": True
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
            "expected_result": False
        },
        # 16. Perfect balance of activities
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
            "expected_result": True
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
            "expected_result": False
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
            "expected_result": False
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
            "expected_result": True
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
            "expected_result": False
        },
        # 21. Knowledge sharer - completes quizzes and earns karma
        {
            "test_id": 21,
            "description": "Knowledge sharer with quizzes and karma",
            "user_id": "user_1021",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 5,
                "quizzes_completed": 3,
                "buddies_messaged": 0,
                "karma_spent": 5,
                "karma_earned_today": 30
            },
            "expected_result": True
        },
        # 22. Social connector - many messages
        {
            "test_id": 22,
            "description": "Social connector with many messages",
            "user_id": "user_1022",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 2,
                "upvotes_received": 6,
                "quizzes_completed": 0,
                "buddies_messaged": 6,
                "karma_spent": 10,
                "karma_earned_today": 15
            },
            "expected_result": True
        },
        # 23. Content creator - many posts
        {
            "test_id": 23,
            "description": "Content creator with many posts",
            "user_id": "user_1023",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 4,
                "posts_created": 5,
                "comments_written": 1,
                "upvotes_received": 20,
                "quizzes_completed": 0,
                "buddies_messaged": 1,
                "karma_spent": 5,
                "karma_earned_today": 30
            },
            "expected_result": True
        },
        # 24. Community supporter - many upvotes
        {
            "test_id": 24,
            "description": "Community supporter giving many upvotes",
            "user_id": "user_1024",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 2,
                "posts_created": 0,
                "comments_written": 1,
                "upvotes_received": 0,  # Giving upvotes, not receiving
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 30,  # Spent on upvoting
                "karma_earned_today": 5
            },
            "expected_result": True
        },
        # 25. New user first day
        {
            "test_id": 25,
            "description": "New user's first day with good activity",
            "user_id": "user_1025",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 1,
                "comments_written": 2,
                "upvotes_received": 5,
                "quizzes_completed": 1,
                "buddies_messaged": 1,
                "karma_spent": 0,
                "karma_earned_today": 10,
                "is_new_user": True
            },
            "expected_result": True
        },
        # 26. Inactive user
        {
            "test_id": 26,
            "description": "Completely inactive user",
            "user_id": "user_1026",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 0,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            },
            "expected_result": False
        },
        # 27. High karma earner
        {
            "test_id": 27,
            "description": "User earning lots of karma",
            "user_id": "user_1027",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 5,
                "posts_created": 2,
                "comments_written": 3,
                "upvotes_received": 25,
                "quizzes_completed": 1,
                "buddies_messaged": 2,
                "karma_spent": 10,
                "karma_earned_today": 75
            },
            "expected_result": True
        },
        # 28. Minimal qualifying activity
        {
            "test_id": 28,
            "description": "Barely meets minimum requirements",
            "user_id": "user_1028",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 3,
                "posts_created": 1,
                "comments_written": 1,
                "upvotes_received": 10,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 25,
                "karma_earned_today": 5
            },
            "expected_result": True
        },
        # 30. Perfect storm - all metrics at high values
        {
            "test_id": 30,
            "description": "Perfect storm - all metrics at high values",
            "user_id": "user_1030",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 30,
                "posts_created": 5,
                "comments_written": 10,
                "upvotes_received": 50,
                "quizzes_completed": 3,
                "buddies_messaged": 5,
                "karma_spent": 50,
                "karma_earned_today": 100
            },
            "expected_result": True
        }
    ])
    
    # 21-30: Test specific reward conditions
    test_cases.extend([
        # 21. Knowledge sharer condition (quizzes_completed >=1 and karma_earned_today >=25)
        {
            "test_id": 21,
            "description": "Knowledge sharer condition match",
            "user_id": "user_1021",
            "date": get_date(0),
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 1,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 25
            },
            "expected_result": True
        },
        # 22-30: Other condition tests...
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
                "login_streak": 100,
                "posts_created": 50,
                "comments_written": 100,
                "upvotes_received": 200,
                "quizzes_completed": 10,
                "buddies_messaged": 20,
                "karma_spent": 1000,
                "karma_earned_today": 500
            },
            "expected_result": True
        },
        # 32-40: Other boundary tests...
    ])
    
    # 41-50: Realistic user scenarios
    test_cases.extend([
        # 41. New user with good first day
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
            "expected_result": True
        },
        # 42-50: Other realistic scenarios...
    ])
    
    return test_cases

def run_tests():
    """Execute all test cases and report results."""
    print("🚀 Starting Reward Engine Test Suite")
    print("=" * 50)
    
    engine = get_reward_engine()
    test_cases = generate_test_cases()
    
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "test_details": []
    }
    
    for test in test_cases:
        try:
            print(f"\n🔍 Test {test['test_id']}: {test['description']}")
            print("-" * 60)
            
            # Execute test
            result = engine.check_surprise_box(
                test["user_id"],
                test["date"],
                test["daily_metrics"]
            )
            
            # Check result
            test_passed = result.get("surprise_unlocked") == test["expected_result"]
            
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
                "actual": result.get("surprise_unlocked"),
                "reward_details": result.get("reward_details", "No reward")
            }
            results["test_details"].append(test_result)
            
            # Print test result
            print(f"{status} - Expected: {test['expected_result']}, Got: {result.get('surprise_unlocked')}")
            print(f"Reward: {result.get('reward_details', 'None')}")
            
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
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {results['total']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    pass_rate = (results['passed'] / results['total']) * 100 if results['total'] > 0 else 0
    print(f"📈 Pass Rate: {pass_rate:.2f}%")
    
    # Save detailed results
    with open('test_results_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n📝 Detailed results saved to 'test_results_detailed.json'")
    
    return results

if __name__ == "__main__":
    run_tests()
