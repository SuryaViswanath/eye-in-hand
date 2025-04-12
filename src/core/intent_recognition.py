import requests
from src.constants.constants import (DEEPSEEK,
                                     INTENT_PREDICTION_PROMPT)


class IntentRecognition:
    """
    Based on the cue information about hand displacement
    and gaze direction, this class generates scene understanding
    and next action that can take place
    """
    def __init__(self, gaze_direction,
                 hand_displacement_direction) -> None:
        self.gaze_direction = gaze_direction
        self.hand_displacement_direction = hand_displacement_direction

    def scene_understanding(self) -> str:
        return f"From the video, and the different cues what we understood\
          from looking at all of the frames are as follows.\
                {self.gaze_direction},\
                    {self.hand_displacement_direction}"

    def final_intent_prediction(self) -> str:
        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": DEEPSEEK,
                                "prompt": INTENT_PREDICTION_PROMPT.format(
                                    gaze_info=self.gaze_direction,
                                    hand_info=self.hand_displacement_direction,
                                    optional_context=""),
                                "stream": False  # True for streaming response
                            })

        # Extract and print the response
        data = response.json()
        return data["response"]
