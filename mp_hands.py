import cv2
import sys
import os
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

codefold = True

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    #CHANGE VIDEO NAME
cap = cv2.VideoCapture('asmrvid3.mov')
array = []
while cap.isOpened():
  success, image = cap.read()
  if not success:
    npArray = np.array(array)
    #print(npArray)
    #sys.stdout = open("output-array-temp.txt", "w")
    #print(array)
    #sys.stdout.close()
    #CHANGE OUTPUT FILE NAME
    #np.savez("output-longvid-NEW-npz", npArray)
    print("******** END ***************")
    break


  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  
  
  if results.multi_hand_landmarks:
    #print("FRAME ***************************************")
    framrArr = []
    numHands = 0
    handTypes = []
    # first check if there's only one hand
    for idx, hand_handedness in enumerate(results.multi_handedness):
        handedness_dict = MessageToDict(hand_handedness)
        handTypes.append(handedness_dict.get("classification")[0].get("label"))
        numHands+=1




    if(numHands == 1 and handTypes[0]=="Right"):
        for x in range(0,21):
            framrArr.append([0,0,0])
    
    print(len(results.multi_handedness))
    for hand_landmarks in results.multi_hand_landmarks:
      if(codefold):
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          ''' ----------------------------HANDS--------------------------------------- '''
          ''' ------------------------------------------------------------------------ '''
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z ])
          
          framrArr.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z ])
          ''' ------------------------------------------------------------------- '''
          ''' ------------------------------------------------------------------------ '''
    if(numHands == 1 and handTypes[0]=="Left"):
        for x in range(0,21):
            framrArr.append([0,0,0])
    print("FRAME^^^^^^^^^^^^")
    #print(framrArr)
    array.append(np.array(framrArr))
    
  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
    
    
            
hands.close()
cap.release()
