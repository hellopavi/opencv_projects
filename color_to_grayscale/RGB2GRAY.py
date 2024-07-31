import cv2
import imutils

#Read the image
nimg = cv2.imread('Nimg.png') 

#covert that image into gray
grayimg = cv2.cvtColor(nimg , cv2.COLOR_BGR2GRAY)

#Convert that grayscale image into black&white using thershold
threshimg = cv2.threshold(grayimg, 140,255, cv2.THRESH_BINARY) [1]
threshimg = cv2.dilate(threshimg , None , iterations = 2)

# resize the screen output
grayimg = imutils.resize(grayimg , width=500)
threshimg = imutils.resize(threshimg , width=500)

#show the image
cv2.imshow('Gray image', grayimg) 
cv2.imshow('Black and white', threshimg)


#Save the image 
cv2.imwrite("Gimg.jpg", grayimg)
cv2.imwrite("b&w.jpg", threshimg)
print('image saved ')

cv2.waitKey(10000) 
cv2.destroyAllWindows()


