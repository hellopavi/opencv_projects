import cv2
from facial_emotion_recognition import EmotionRecognition


er = EmotionRecognition(device = 'cpu')


vidpath = 'emotion.mp4'
cam = cv2.VideoCapture(vidpath)

while True :
    a ,frame = cam.read()
    frame = er.recognise_emotion(frame , return_type = 'BGR')
    text = 'To quit Press : Esc'
    cv2.putText(frame ,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX , 0.5, (102,102,0),2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27 :
        break
    
cam.release()
cv2.destroyAllWindows()
