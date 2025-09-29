import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
# Constants
SRC_POINTS = np.array([[261, 310], [261, 692], [883, 692], [883, 310]], dtype=np.float32)
DST_SIZE = (1280, 720)
DST_POINTS = np.array([[0, 0], [0, DST_SIZE[1]], [DST_SIZE[0], DST_SIZE[1]], [DST_SIZE[0], 0]], dtype=np.float32)
MAX_HOMOGRAPHY_HISTORY = 10  # Limit homography list length

# Homography history (for KNN averaging)
homography_history = deque(maxlen=MAX_HOMOGRAPHY_HISTORY)

# Read video
cap = cv2.VideoCapture("/Users/nguyenconghung/Documents/Video/ConveyerBelt/Untitled.mov")
ret, prev_frame = cap.read()

if not ret:
    print("Failed to read from video.")
    cap.release()
    exit()

# Apply perspective transform to initial frame
prev_rImage = cv2.warpPerspective(prev_frame, cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS), DST_SIZE)

# Convert to grayscale for optical flow
prev_gray = cv2.cvtColor(prev_rImage, cv2.COLOR_BGR2GRAY)

prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1, qualityLevel=0.3, minDistance=7)

plt.ion()

# Time tracking
prev_time = time.time()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (DST_SIZE[0], DST_SIZE[1]))
pre_speed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perspective transform
    rImage = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS), DST_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(rImage, cv2.COLOR_BGR2GRAY)

    # Optical flow to track motion
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    if next_pts is not None :
        # Calculate pixel displacement
        dx = next_pts[0][0][0] - prev_pts[0][0][0]
        dy = next_pts[0][0][1] - prev_pts[0][0][1]
        displacement = (dx**2 + dy**2) ** 0.5

        # Time delta
        current_time = time.time()
        dt = current_time - prev_time

        # Speed in pixels per second
        speed = displacement / dt
        # print(f"Speed: {speed:.2f} pixels/s")

        # Update for next iteration
        prev_gray = gray.copy()
        prev_pts = next_pts
        prev_time = current_time
        

    point = tuple(int(x) for x in next_pts[0][0])
    if point[0] < 0:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1, qualityLevel=0.3, minDistance=7)
    # cv2.circle(rImage, point, 5, (0, 255, 0), -1)
    # # write speed on the image
    # if (pre_speed > speed):
    #     cv2.putText(rImage, f"Speed: {pre_speed:.2f} pixels/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # else:
    #     pre_speed = speed
    # cv2.imshow("Tracking", rImage)

    out.write(rImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
out.release()