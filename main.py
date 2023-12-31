# Importing Required Packages
import cv2
from random import randrange

# Importing Face Data's XML
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Selecting Image
# img = cv2.imread('RadheKrishna.jpg') # imread Stands For ImageRead

# Getting WebCam
webcam = cv2.VideoCapture(0)

# Getting Frames
while True:
  # Reading Current Frame
  successful_frame_read, frame=webcam.read()

  # Converting Into GreyScale
  greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Detecting Faces
  face_coordinates = trained_face_data.detectMultiScale(frame)

  # Drawing Rectangles Over Faces
  for (x,y,w,h) in face_coordinates:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(0, 255), randrange(0, 255), randrange(0, 255)), 2)

  # PopUp Window Configuration
  cv2.imshow('Face Detector', frame)
  cv2.waitKey(1) # Pauses Window

  print(face_coordinates)

# Confirmatory Code
print("The Code Has Completed Successfully.")