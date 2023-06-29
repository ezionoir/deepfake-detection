import json
import numpy as np
import cv2

def load_landmarks(path):
    with open(path, 'r') as f:
        return np.array(json.load(f))
    
def get_frame(path, index=0):
    video = cv2.VideoCapture(path)

    frame_count = 0
    success, frame = video.read()

    while frame_count < index:
        success, frame = video.read()
        if not success:
            break
        frame_count += 1

    return frame