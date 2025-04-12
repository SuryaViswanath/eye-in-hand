# Required packages: pip install opencv-python mediapipe
# Note: The linter may show errors for these imports, but they will work if installed
import cv2
import mediapipe as mp
from src.core.user_cue_identifiers import UserCueIdentifiers
from src.core.intent_recognition import IntentRecognition
from collections import deque, Counter


wrist_history = deque(maxlen=5)
head_history = deque(maxlen=5)

gaze_direction = ""
hand_displacement = ""
last_known_hand_displacement = "Center"  # Initialize with a default value


class Object:
    def __init__(self, name, region) -> None:
        self.name = name
        self.region = region


objects = [
    Object("Left Object", (0.0, 0.0, 0.3, 1.0)),    # Left third of screen
    Object("Center Object", (0.3, 0.0, 0.7, 1.0)),  # Center third
    Object("Right Object", (0.7, 0.0, 1.0, 1.0))    # Right third
]

# start video
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

mp_holistic = mp.solutions.holistic

# uci = UserCueIdentifiers()
# ir = IntentRecognition()

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # extracting the required landmarks from the frame
        face_landmarks = results.face_landmarks
        left_hand = results.left_hand_landmarks
        right_hand = results.right_hand_landmarks

        # keep active hand for tracking
        active_hand = right_hand or left_hand

        uci = UserCueIdentifiers(face_landmarks=face_landmarks,
                                 hand_landmarks=active_hand)

        if active_hand:
            wrist = active_hand.landmark[0]

        # Get hand displacement and gaze direction
        current_hand_displacement = uci.identify_hand_displacement()
        if "No movement" not in current_hand_displacement:
            last_known_hand_displacement = current_hand_displacement
            hand_displacement = current_hand_displacement
        else:
            hand_displacement = last_known_hand_displacement

        if face_landmarks:
            gaze_direction = uci.identify_gaze_direction()
        else:
            gaze_direction = "No face detected"

        # Display the frame
        cv2.imshow('Eye-in-Hand Tracking', frame)

        # Wait for key press and check for ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("ESC key pressed, exiting...")
            break
        elif key != 255:  # If any other key is pressed
            print(f"Key pressed: {key}")

    print("From the video footage, the direction of the gaze Observed: "
          , gaze_direction, " and the direction of the hand displacement is: "
          , hand_displacement)
    ir = IntentRecognition(gaze_direction, hand_displacement)
    final_intent_prediction = ir.final_intent_prediction()
    print("The final intent prediction is: \n\n", final_intent_prediction)

cap.release()
cv2.destroyAllWindows()



#  if __name__ == "__main__":
#     # Initialize MediaPipe Holistic
#     mp_holistic = mp.solutions.holistic
#     holistic = mp_holistic.Holistic(
#         min_detection_confidence=0.5, 
#         min_tracking_confidence=0.5
#     )
    
#     # Try different camera indices
#     camera_index = 0
#     print(f"Trying to open camera with index {camera_index}...")
#     cap = cv2.VideoCapture(camera_index)
    
#     # Check if camera opened successfully
#     if not cap.isOpened():
#         print(f"Failed to open camera with index {camera_index}")
#         # Try another camera index
#         camera_index = 1
#         print(f"Trying to open camera with index {camera_index}...")
#         cap = cv2.VideoCapture(camera_index)
        
#         if not cap.isOpened():
#             print(f"Failed to open camera with index {camera_index}")
#             print("Could not open any camera. Please check your camera connection.")
#             exit(1)
    
#     print(f"Successfully opened camera with index {camera_index}")
    
#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to capture frame")
#                 break

#             # Convert to RGB for MediaPipe
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)

#             # Extract landmarks
#             face_landmarks = (results.face_landmarks.landmark
#                               if results.face_landmarks else None)
#             left_hand = results.left_hand_landmarks
#             right_hand = results.right_hand_landmarks

#             # Combine both hands, prefer the one that is visible
#             active_hand = right_hand or left_hand  # Right hand has priority

#             # Initialize UserCueIdentifiers with the landmarks
#             uci = UserCueIdentifiers(face_landmarks, active_hand)

#             # Get gaze direction and hand displacement
#             gaze_direction = uci.identify_gaze_direction()
#             hand_displacement = uci.identify_hand_displacement()

#             # Initialize IntentRecognition with the cues
#             ir = IntentRecognition(gaze_direction, hand_displacement)

#             # Get scene understanding
#             scene_understanding = ir.scene_understanding()
#             print(f"Scene Understanding: {scene_understanding}")

#             # Get final intent prediction
#             intent_prediction = ir.final_intent_prediction()
#             print(f"Intent Prediction: {intent_prediction}")

#             # Display the frame with text overlay
#             # Add text to the frame
#             cv2.putText(frame, gaze_direction, (10, 30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, hand_displacement, (10, 70), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             # Display the frame
#             cv2.imshow('Eye-in-Hand Tracking', frame)

#             # Break the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         # Release resources
#         cap.release()
#         cv2.destroyAllWindows()
#         holistic.close()
