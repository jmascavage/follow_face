import cv2

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
    print("draw")
    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Show the frame
    print("showing image")
    cv2.imshow('Face Tracking', frame)

    # Exit on 'q'
    print("check for q")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
