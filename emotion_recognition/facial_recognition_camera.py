import cv2
from facial_emotion_recognition import EmotionRecognition

#Load the EmotionRecognition library
er = EmotionRecognition(device = 'cpu')

#set the camera input
cam = cv2.VideoCapture(0)

#Read the camera frame by frame and load that into algorithm and show the output
while True :
    a ,frame = cam.read()
    frame = er.recognise_emotion(frame , return_type = 'BGR')
    text = 'To quit Press : Esc'
    cv2.putText(frame ,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX , 0.5, (102,102,0),2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27 :
        break

#Relase the camera   
cam.release()
cv2.destroyAllWindows()
