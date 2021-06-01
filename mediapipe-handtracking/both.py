import cv2
import sys
import os
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict

from mediapipe.python._framework_bindings import _packet_getter
from mediapipe.python._framework_bindings import packet as mp_packet


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# initialize hands with detection and tracking confidence levels
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
    

# change 'tapvid.mkv' to name of your video file
cap = cv2.VideoCapture('tapvid.mkv')


allFrames = [] # array of all frames in the video
frameCount = 0 # keeps track of current frame being iterated
confidenceScores = [] # array of confidence scores


while cap.isOpened():

  success, image = cap.read()

  # when the video ends:
  if not success:

    npArray = np.array(allFrames) # save allFrames as a numpy array
    npArrayConfScores = np.array(confidenceScores) # save confidenceScores as a numpy array

    np.savez("output-landmarks", npArray) # save allFrames numpy array to .npz file
    np.savez("output-confidence-scores", npArrayConfScores) # save confidenceScores numpy array to .npz file

    print("******** END ***************")
    break


  # image processing 
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  if results.multi_hand_landmarks:

    currentFrame = [] # stores landmarks of both hands of current frames
    numHands = 0 # initially the number of hands detected is zero 
    handTypes = [] # used later to store left or right hand labels

    # check how many hands there are
    for idx, hand_handedness in enumerate(results.multi_handedness):
        handedness_dict = MessageToDict(hand_handedness)
        handTypes.append(handedness_dict.get("classification")[0].get("label"))
        numHands+=1

    # if there's only a right hand detected, add 21 [0,0,0] arrays to the current frame
    # this is a placeholder for a missing left hand
    
    if(numHands == 1 and handTypes[0]=="Right"):
        for x in range(0,21):
            currentFrame.append([0,0,0])
    
    # iterate through each hand 
    for hand_landmarks in results.multi_hand_landmarks:

          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

          # for all 21 hand landmarks, add their xyz coordinates to the current frame
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z ])
          
          currentFrame.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z ])
    
    # at the end, check if the only hand you had was a left hand
    # if so, add 21 arrays of [0,0,0] as a placeholder for the missing right hand
    if(numHands == 1 and handTypes[0]=="Left"):
        for x in range(0,21):
            currentFrame.append([0,0,0])

    allFrames.append(np.array(currentFrame)) # add the current frame of hand landmarks to the array

    frameCount+=1 # corresponds to the index of values of currentFrame at which the palm detection score has changed

    # CONFIDENCE SCORES
    # confidence scores: [ [frameCount, [confScoreHand1, confScoreHand2]] ]
    # first check to see if there has been a change in the confidence scores
    if results.palm_detections:
    	temp = []
    	for palm_detection in results.palm_detections:
    		temp.append(palm_detection.score)
    	confidenceScores.append(np.array([frameCount,np.array(temp)])) # add the frame# and the array of confidence scores to the confidenceScores array
    
  # cv2.imshow('MediaPipe Hands', image)
  # use the above line if you want to display the hand landmark annotations on the screen
  # not advised if running on remote (you may run into errors since it is generally not supported on remote)

  if cv2.waitKey(5) & 0xFF == 27:
    break
            
hands.close()
cap.release()
