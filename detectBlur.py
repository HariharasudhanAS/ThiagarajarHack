import numpy as np
import cv2

# Initialize face cascade from file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set path to video
cap = cv2.VideoCapture('/home/work/Downloads/blurr sample.mp4')
frame_count = 0
c = 0

# Loop runs when the video is open
while(cap.isOpened()):
    
    ret, frame = cap.read()
    # Continue if no frame is read
    if ret != True:
        continue
    
    # Maintain frame counter
    frame_count += 1
    
    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(150, 150))
    
    # Check for number of faces detected in frame
    recog = np.shape(faces)[0]
    
    # Increment c if no face is found
    if recog == 0:
        c += 1
    
    
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        # If a face recognised after no face in the last 20 or more frames
        if c > 20:
            print("Frame number ", frame_count-c, " to ", frame_count, end='')
            c = 0
    
    cv2.putText(frame, "Number of faces detected: " + str(np.shape(faces)[0]), 
                (0,np.shape(faces)[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 10,  (0,0,0), 1)
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
