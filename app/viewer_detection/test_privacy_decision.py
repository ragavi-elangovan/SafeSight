from app.viewer_detection.privacy_decision_engine import (
    PrivacyDecisionEngine
)


# ============================================================
# CREATE ENGINE
# ============================================================

engine = PrivacyDecisionEngine()


# ============================================================
# TEST CASES
# ============================================================

test_cases = [

    {
        "name": "Authorized - Normal Distance",
        "identity": "RAGAVI",
        "match_score": 0.81,
        "proximity": "FAR",
        "distance_cm": 130,
        "head_direction": "FORWARD",
        "gaze_direction": "CENTER",
    },

    {
        "name": "Authorized - Very Close",
        "identity": "RAGAVI",
        "match_score": 0.81,
        "proximity": "VERY CLOSE",
        "distance_cm": 32,
        "head_direction": "FORWARD",
        "gaze_direction": "CENTER",
    },

    {
        "name": "Unknown - Far",
        "identity": "UNKNOWN",
        "match_score": 0.32,
        "proximity": "FAR",
        "distance_cm": 150,
        "head_direction": "LEFT",
        "gaze_direction": "LEFT",
    },

    {
        "name": "Unknown - Medium - Looking",
        "identity": "UNKNOWN",
        "match_score": 0.32,
        "proximity": "MEDIUM",
        "distance_cm": 90,
        "head_direction": "FORWARD",
        "gaze_direction": "CENTER",
    },

    {
        "name": "Unknown - Medium - Not Looking",
        "identity": "UNKNOWN",
        "match_score": 0.32,
        "proximity": "MEDIUM",
        "distance_cm": 90,
        "head_direction": "LEFT",
        "gaze_direction": "LEFT",
    },

    {
        "name": "Unknown - Close",
        "identity": "UNKNOWN",
        "match_score": 0.32,
        "proximity": "CLOSE",
        "distance_cm": 55,
        "head_direction": "FORWARD",
        "gaze_direction": "CENTER",
    },

    {
        "name": "Unknown - Very Close",
        "identity": "UNKNOWN",
        "match_score": 0.32,
        "proximity": "VERY CLOSE",
        "distance_cm": 30,
        "head_direction": "FORWARD",
        "gaze_direction": "CENTER",
    },
]


# ============================================================
# RUN TESTS
# ============================================================

print()
print("=" * 70)
print("SafeSight - PRIVACY DECISION ENGINE TEST")
print("=" * 70)


for test in test_cases:

    result = engine.decide(
        identity=test["identity"],
        match_score=test["match_score"],
        proximity=test["proximity"],
        distance_cm=test["distance_cm"],
        head_direction=test["head_direction"],
        gaze_direction=test["gaze_direction"],
    )


    print()
    print(f"Test     : {test['name']}")
    print(f"Identity : {test['identity']}")
    print(f"Distance : {test['distance_cm']} cm")
    print(f"Proximity: {test['proximity']}")
    print(f"Head     : {test['head_direction']}")
    print(f"Gaze     : {test['gaze_direction']}")
    print(f"ACTION   : {result.action}")
    print(f"Risk     : {result.risk_score:.2f}")
    print(f"Reason   : {result.reason}")


print()
print("=" * 70)
print("Privacy Decision Engine test completed.")
print("=" * 70)