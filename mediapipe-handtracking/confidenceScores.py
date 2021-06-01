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


frameCount = 0 # keeps track of current frame being iterated
confidenceScores = [] # array of confidence scores


while cap.isOpened():

  success, image = cap.read()

  # when the video ends:
  if not success:

    npArrayConfScores = np.array(confidenceScores) # save confidenceScores as a numpy array
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
