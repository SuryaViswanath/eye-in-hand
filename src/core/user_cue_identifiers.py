import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
from src.constants.constants import (NOSE,
                                    LEFT_EYE,
                                     RIGHT_EYE
)
from collections import deque, Counter

mp_holistic = mp.solutions.holistic
wrist_movement_history = deque(maxlen=5)


class UserCueIdentifiers:
    """
    This class handles the identification of different cues
    from the video footage. This returns scene understanding
    for the LLMs to infer what is happening
    """
    def __init__(self, face_landmarks, hand_landmarks) -> None:
        self.face_landmarks = face_landmarks
        self.hand_landmarks = hand_landmarks

    def identify_gaze_direction(self) -> str:
        # Check if face landmarks are available
        if self.face_landmarks is None:
            return "No face detected"
            
        gaze_direction = ""
        nose = self.face_landmarks.landmark[NOSE]
        left_eye = self.face_landmarks.landmark[LEFT_EYE]
        right_eye = self.face_landmarks.landmark[RIGHT_EYE]

        # eye line using pin hole camera strategy
        eye_line_width = left_eye.x - right_eye.x

        nose_position = (nose.x - right_eye.x)/ eye_line_width

        if nose_position < 0.4:
            gaze_direction = "Left"
        elif nose_position > 0.6:
            gaze_direction = "Right"
        else:
            gaze_direction = "Center"
        return f"Seems like the person in the frame is looking\
          towards {gaze_direction}"

    def identify_hand_displacement(self) -> str:
        hand_displacement = ""

        if self.hand_landmarks is None:
            return "No movement"

        wrist = self.hand_landmarks.landmark[0]
        wrist_movement_history.append(wrist.x)

        if len(wrist_movement_history) < 3:
            return "Center"

        # what is the hand displacement like
        delta = wrist_movement_history[-1] - wrist_movement_history[0]

        movement_threshold = 0.02 

        if delta > movement_threshold:
            hand_displacement = "Left"

        elif delta < -movement_threshold:
            hand_displacement = "Right"

        else:
            hand_displacement = "Center"

        return f"""Seems like the hand of the person in the frame
              is moving towards {hand_displacement}"""

    def identify_hand_shape(self) -> str:
        hand_shape = ""
        return f"Seems like the shape of the hand of the\
              person is {hand_shape}"
