import cv2
import imutils

# Define video path
vidpath = 'testvid.mp4'

# Open video capture object
cam = cv2.VideoCapture(vidpath)

# Get the frame rate of the input video
fps = cam.get(cv2.CAP_PROP_FPS)

# Get the width and height of the frames (after resizing)
frame_width = 1000
#frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) * (frame_width / int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))))

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
#out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Define initial variables
firstframe = None
area = 500

while True:
    # Reading the camera input from cam
    ret, img = cam.read()

    if not ret:
        break
    
    text = 'Normal'

    # Resizing the output frame
    img_resized = imutils.resize(img, width=frame_width)

    # Convert that video frame into grayscale
    grayimg = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Smoothening that grayscale image
    gausimg = cv2.GaussianBlur(grayimg, (21, 21), 0)

    # Capturing the firstframe of the video capture
    if firstframe is None:
        firstframe = gausimg
        continue

    # Find the absolute difference from firstframe with current frame
    imgdiff = cv2.absdiff(firstframe, gausimg)

    # Convert that grayscale image into black & white using threshold
    threshimg = cv2.threshold(imgdiff, 50, 255, cv2.THRESH_BINARY)[1]
    threshimg = cv2.dilate(threshimg, None, iterations=2)

    # Finding contours from the threshold image
    cont = cv2.findContours(threshimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)

    # Detect and mark moving objects with bounding boxes and text
    for c in cont:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = 'Moving object detected'

    print(text)
    info = 'To quit the video: press Q'
    # Display frame with text information
    cv2.putText(img_resized, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img_resized, info, (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 102, 0), 2)

    # Write the frame to the output video
    #out.write(img_resized)

    # Display the frame
    cv2.imshow('camerafeed', img_resized)

    # Handle user input (press 'q' to quit)
    key = cv2.waitKey(int(1000 / fps))  # Adjust delay based on frame rate
    if key == ord('q'):
        print("Exit")
        break

# Release resources
cam.release()
#out.release()
cv2.destroyAllWindows()
