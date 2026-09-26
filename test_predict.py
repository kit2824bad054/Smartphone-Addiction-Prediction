import requests
import json

test_cases = [
    {
        "name": "Low-risk case",
        "payload": {
            "screen_time_hours": 1.5,
            "unlocks_per_day": 20,
            "social_media_hours": 0.5,
            "night_usage_ratio": 0.05,
            "sleep_hours": 8.5
        }
    },
    {
        "name": "Moderate-risk case",
        "payload": {
            "screen_time_hours": 5.0,
            "unlocks_per_day": 80,
            "social_media_hours": 2.5,
            "night_usage_ratio": 0.3,
            "sleep_hours": 6.5
        }
    },
    {
        "name": "High-risk case",
        "payload": {
            "screen_time_hours": 11.0,
            "unlocks_per_day": 200,
            "social_media_hours": 7.0,
            "night_usage_ratio": 0.7,
            "sleep_hours": 4.0
        }
    }
]

url = "http://127.0.0.1:8000/predict"

for tc in test_cases:
    print("=" * 70)
    print(f"TEST CASE: {tc['name']}")
    print("PAYLOAD:")
    print(json.dumps(tc['payload'], indent=2))
    
    response = requests.post(url, json=tc['payload'])
    print(f"STATUS CODE: {response.status_code}")
    
    data = response.json()
    print("RESPONSE JSON:")
    print(json.dumps(data, indent=2))
    print(f"VERIFICATION: Result={data['risk_level']} | Score={data['risk_score']} | Match={tc['name'].startswith(data['risk_level'])}")

print("=" * 70)
