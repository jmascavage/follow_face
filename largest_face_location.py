import cv2
import time

# Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the USB camera
cap = cv2.VideoCapture(0)  # Use 0 for the default USB cam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (Haar works on gray images)

    print("converting to grayscale.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    print("detecting faces")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around faces
 #   print("draw")
 #   for (x, y, w, h) in faces:
 #       center_x = x + w // 2
 #       center_y = y + h // 2
 #       cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
 #       cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Initialize variables
    largest_face = None
    max_area = 0

    # Find the largest face
    print("finding largest face")
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)

    # If a face is found, process it
    if largest_face is not None:
        print("face found = processing it...")
        x, y, w, h = largest_face
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Draw rectangle and center point
        print("draw rectangle around and circle center")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)
        
        # Show the frame
        print("showing image")
        cv2.imshow('Face Tracking', frame)


        # Calculate frame center
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        # Show offset (can use this to move servos later)
        offset_x = face_center_x - frame_center_x
        offset_y = face_center_y - frame_center_y

        print(f"Face offset: X={offset_x}, Y={offset_y}")
        
    # Exit on 'q'
    print("check for q")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
