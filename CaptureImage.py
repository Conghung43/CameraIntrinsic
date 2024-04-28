import cv2

def main():
    # Open the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(1)
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            continue

        # Display the resulting frame
        cv2.imshow('Capture Image', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Capture image when 'c' is pressed
        if key == ord('c'):
            cv2.imwrite(f'images/CameraCapture/{count}.jpg', frame)
            print("Image captured!")
            count += 1
        
        # Quit when 'q' is pressed
        elif key == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
