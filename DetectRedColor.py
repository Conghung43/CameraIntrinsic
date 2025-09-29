import cv2
import numpy as np
import time

# Constants
WINDOW_NAME = "Display"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
RED_IMAGE_SIZE = 100
DETECTION_THRESHOLD = 500  # Number of red pixels needed for detection

def create_blank_frame():
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

def draw_red_square(frame):
    center_x, center_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    top_left = (center_x - RED_IMAGE_SIZE // 2, center_y - RED_IMAGE_SIZE // 2)
    bottom_right = (center_x + RED_IMAGE_SIZE // 2, center_y + RED_IMAGE_SIZE // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)
    return frame

def detect_red_color(frame):
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_red1 = np.array([0, 100, 100])
    # upper_red1 = np.array([10, 255, 255])
    # lower_red2 = np.array([160, 100, 100])
    # upper_red2 = np.array([180, 255, 255])
    # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # red_mask = mask1 | mask2
    # red_pixels = cv2.countNonZero(red_mask)
    # return red_pixels > DETECTION_THRESHOLD
    return frame[106,247][2] > 200

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera")
        return

    cv2.namedWindow(WINDOW_NAME)
    response_times = []

    for i in range(4):
        print(f"Trial {i+1}...")

        # Step 0: Show blank screen
        blank_frame = create_blank_frame()
        cv2.imshow(WINDOW_NAME, blank_frame)
        cv2.waitKey(500)

        # Step 1: Show red image in center
        red_frame = draw_red_square(create_blank_frame())
        cv2.imshow(WINDOW_NAME, red_frame)
        cv2.waitKey(1)  # show immediately

        # Step 2: Start timing and detect red from camera
        start_time = time.time()
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while True:
            ret, cam_frame = cap.read()
            if not ret:
                continue
            # get top-right quarter of the frame
            frame = cam_frame[(int)(height/6):(int)(height/3), (int)(width/6) :(int)(width/3)]
            cv2.imwrite("test.jpg", frame)
            if detect_red_color(frame):
                elapsed = time.time() - start_time
                response_times.append(elapsed)
                print(f"Detected red in {elapsed:.3f} seconds")

                # Step 3: Turn off red image for 1 second
                cv2.imshow(WINDOW_NAME, blank_frame)
                cv2.waitKey(1)
                time.sleep(1)
                break

    cap.release()
    cv2.destroyAllWindows()

    if response_times:
        avg = sum(response_times) / len(response_times)
        print(f"\nAverage detection time: {avg:.3f} seconds")

if __name__ == "__main__":
    main()
