import cv2
from facial_emotion_recognition import EmotionRecognition

#Load the emotion recognition library
er = EmotionRecognition(device = 'cpu')

#Set image path and read the image
imgpath = 'face_emotion.png'
readimg = cv2.imread(imgpath)

#give the input into emotion recognition algorithm
frame = er.recognise_emotion(readimg , return_type = 'BGR')
cv2.imshow("Frame",frame)

#if you want save the image uncomment this below comment
#cv2.imwrite('expression.jpg',frame)

#Show output for 10 seconds = 10000 milliseconds
cv2.waitKey(10000)
cv2.destroyAllWindows()
