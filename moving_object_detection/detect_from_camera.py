import cv2
import imutils

cam = cv2.VideoCapture(0)

firstframe = None
area = 500

while True :
    #Reading the camera input from cam
    _ , img = cam.read()
    text='Normal'

    
    #Resizing the output frame
    img = imutils.resize(img , width= 1000)

    #Convert that video frame into grayscale
    grayimg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    #Smoothening that grayscale image
    gausimg = cv2.GaussianBlur(grayimg , (21,21), 0)

    #Capturing the firstframe of the video capture
    if firstframe is None :
        firstframe = gausimg
        continue

    #find the absolute difference from firstframe with current frame
    imgdiff = cv2.absdiff(firstframe , gausimg)

    #Convert that grayscale image into black&white using thershold
    threshimg = cv2.threshold(imgdiff, 50,255, cv2.THRESH_BINARY) [1]
    threshimg = cv2.dilate(threshimg , None , iterations = 2)

    #Finding contours from the threshold image
    cont = cv2.findContours(threshimg.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)

     #capture the countours with area condition and create a rectangle with print text

    for c in cont :       
        if cv2.contourArea(c) < area :
             continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img , (x,y),(x+w,y+h),(0,255,0),2)
        text = 'Moving object detected'

    print(text)
    info = 'To quit the video : press Q '
    cv2.putText(img, text,(10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2)
    cv2.putText(img, info,(250,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(102,102,0),2)
    
    cv2.imshow('camerafeed' , img)

    key = cv2.waitKey(10)
    if key == ord('q'):
        print("Exit")
        break

cam.release()
cv2.destroyAllWindows()

    
