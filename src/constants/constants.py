HISTORY_LENGTH = 10

# Threshold for decision stability
DECISION_THRESHOLD = 6  # Out of HISTORY_LENGTH frames, at least this many must agree

# Hand shape classification thresholds
FINGER_TIP_IDS = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
FINGER_PIP_IDS = [6, 10, 14, 18]  # Second joint of each finger
THUMB_TIP_ID = 4
THUMB_IP_ID = 3

NOSE = 1
LEFT_EYE = 33
RIGHT_EYE = 263

DEEPSEEK = "deepseek-r1:1.5b"

INTENT_PREDICTION_PROMPT= """
You are an intelligent reasoning system. Based on behavioral cues from a person in a video, infer whether they are more likely trying to pick the item on the left or the item on the right.

Here are the observed cues:
- Gaze Direction: {gaze_info}
- Hand Displacement: {hand_info}
- Contextual Info (if available): {optional_context}

Based on these cues, which item is the person most likely trying to pick?

Respond with ONLY ONE of the following options:
- "The person is most likely to pick the item on the Left"
- "The person is most likely to pick the item on the Right"
"""
