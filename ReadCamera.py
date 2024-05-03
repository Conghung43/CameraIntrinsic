import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the first camera, 1 for the second, and so on

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

# Loop to continuously read frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
