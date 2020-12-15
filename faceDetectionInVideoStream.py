import cv2, os
import numpy as np
import face_recognition

# Put the face pictures in this folder
# Which you want to detect later
path = "pictures"

#This is just image parsing
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(myList)
print(classNames)

def findEncodings(imgages):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")
print("#"*50)

##########################################################
#IMPORTANT
# I am using a third-party app to connect my phone's camera to my computer
# It's called "DroidCamApp"
# Which connects your phone camera to your computer via WI-FI or USB connection
# So that's why cv2.VideoCapture's CAM-ID is '1' below.
# If you want to use your default camera on your computer
# USE '0' instead.
# ex: cap = cv2.VideoCapture(0)

#This way I can use my phone's camera and show everything to see if
#face detection works. Open Google Images and show pictures to your camera.
##########################################################
cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDistance) ########################THE LOWER IS THE BETTER MATCH

        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 *4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name ,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
