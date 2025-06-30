import requests
import json
import time
from pprint import pprint

BASE_URL = "http://localhost:8000"

def test_health():
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print("Response:", response.json())
    return response.status_code == 200

def test_get_config():
    print("\n=== Testing Get Config ===")
    response = requests.get(f"{BASE_URL}/config")
    print(f"Status Code: {response.status_code}")
    print("Config:")
    pprint(response.json())
    return response.status_code == 200

def test_check_surprise_box():
    print("\n=== Testing Check Surprise Box ===")
    test_data = {
        "user_id": "test_user_123",
        "date": "2024-06-09",
        "daily_metrics": {
            "login_streak": 5,
            "posts_created": 2,
            "comments_written": 3,
            "upvotes_received": 10,
            "quizzes_completed": 1,
            "buddies_messaged": 2,
            "karma_spent": 35,
            "karma_earned_today": 50
        }
    }
    response = requests.post(f"{BASE_URL}/check-surprise-box", json=test_data)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    pprint(response.json())
    return response.status_code == 200

def test_add_rule():
    print("\n=== Testing Add Rule ===")
    rule_data = {
        "conditions": ["quizzes_completed > 0"],
        "description": "Completed a quiz"
    }
    response = requests.post(
        f"{BASE_URL}/rules/quiz_completion",
        json=rule_data
    )
    print(f"Status Code: {response.status_code}")
    print("Response:", response.json())
    return response.status_code in (200, 201)

def test_list_rules():
    print("\n=== Testing List Rules ===")
    response = requests.get(f"{BASE_URL}/rules")
    print(f"Status Code: {response.status_code}")
    print("Rules:")
    pprint(response.json())
    return response.status_code == 200

def test_update_config():
    print("\n=== Testing Update Config ===")
    update_data = {
        "reward_probability_threshold": 0.6,
        "karma_min": 20,
        "karma_max": 2000
    }
    response = requests.patch(f"{BASE_URL}/config", json=update_data)
    print(f"Status Code: {response.status_code}")
    print("Updated Config:")
    pprint(response.json())
    return response.status_code == 200

def run_tests():
    tests = [
        test_health,
        test_get_config,
        test_check_surprise_box,
        test_add_rule,
        test_list_rules,
        test_update_config,
    ]
    
    print("=== Starting API Tests ===")
    results = []
    for test in tests:
        try:
            success = test()
            results.append((test.__name__, success))
        except Exception as e:
            print(f"Error in {test.__name__}: {str(e)}")
            results.append((test.__name__, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
    
    # Check if all tests passed
    all_passed = all(success for _, success in results)
    print("\nAll tests passed!" if all_passed else "\nSome tests failed!")
    return 0 if all_passed else 1

if __name__ == "__main__":
    # Give the server a moment to start
    time.sleep(2)
    exit(run_tests())
