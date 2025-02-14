# Importing Libraries 
import cv2 
import mediapipe as mp 
from math import hypot 
import numpy as np 
import pycaw.pycaw as pycaw
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands 
hands = mpHands.Hands(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2) 
Draw = mp.solutions.drawing_utils 

# Start capturing video from webcam 
cap = cv2.VideoCapture(0) 

while True: 
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) 
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    Process = hands.process(frameRGB) 
    landmarkList = [] 

    if Process.multi_hand_landmarks: 
        for handlm in Process.multi_hand_landmarks: 
            for _id, landmarks in enumerate(handlm.landmark): 
                height, width, _ = frame.shape
                cx, cy = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, cx, cy])

            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS) 

        if len(landmarkList) >= 8:
            x1, y1 = landmarkList[4][1], landmarkList[4][2]  # Thumb tip
            x2, y2 = landmarkList[8][1], landmarkList[8][2]  # Index finger tip
            distance = hypot(x2 - x1, y2 - y1)
            
            # Normalize distance to volume range
            min_dist, max_dist = 20, 200  # Adjust these values if necessary
            min_vol, max_vol = volume.GetVolumeRange()[:2]
            new_vol = np.interp(distance, [min_dist, max_dist], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(new_vol, None)

            # Display volume level on screen
            vol_percentage = np.interp(new_vol, [min_vol, max_vol], [0, 100])
            cv2.putText(frame, f'Volume: {int(vol_percentage)}%', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Hand Gesture Volume Control", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
