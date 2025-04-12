"""Utility functions for drawing visualizations on camera frames."""
import cv2


def draw_objects(frame, objects):
    """Draw object regions on the frame"""
    h, w, _ = frame.shape
    for obj in objects:
        x1, y1, x2, y2 = obj.region
        # Convert normalized coordinates to pixel values
        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)
        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, obj.name, (x1 + 10, y1 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame
