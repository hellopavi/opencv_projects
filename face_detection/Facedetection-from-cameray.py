import cv2
import imutils

# Open the video capture object
cam = cv2.VideoCapture(0)

# Path to the face detection algorithm (XML file)
algo = 'haarcascade_frontalface_default.xml'

# Load the face detection cascade classifier
loadalgo = cv2.CascadeClassifier(algo)

# Main loop for video processing
while True :

    # Capture a frame from the video
    _,img = cam.read()

    # Check if frame capture was successful
    if img is None :
        print('video ended')
        break
    
    # Resize the frame
    img = imutils.resize(img , width=700)

    # Convert the frame to grayscale for face detection
    grayimg = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)

    # Detect faces using the loaded cascade classifier
    face = loadalgo.detectMultiScale(grayimg , scaleFactor=1.3 ,minNeighbors=4)

    # Draw a green rectangle around the detected face
    for (x, y, w, h) in face :
        cv2.rectangle(img , (x,y),(x+w , y+h),(0,255,0),5)

    # Display information for quitting the video
    info = 'To quit the video : press Q '
    cv2.putText(img, info,(250,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(102,102,0),2)

    # Display the frame with detected faces
    cv2.imshow('Face detection',img)

    # Wait for a key press for 10 milliseconds
    key = cv2.waitKey(10)

    # If 'q' key is pressed, exit the loop
    if key == ord('q'):
        break

# Releas the video and close the window
cam.release()
cv2.destroyAllWindows()
        
