import cv2
import numpy as np
import time

# Parameters
INIT_CROP_SCALE = 0.5     # Initial center crop when tracking_roi is None
DETECTION_MARGIN = 100    # Larger margin when searching (tracking_roi is None)
REFINEMENT_MARGIN = 25    # Smaller margin when tracking (tracking_roi is not None)
MIN_MATCH_COUNT = 10      # Minimum ORB matches (after ratio test)
ORB_FEATURES = 700       # Number of ORB features to detect (adjust this!)
RATIO_TEST_THRESHOLD = 0.75 # Threshold for Lowe's ratio test

# Load query image
query_img = cv2.imread("Images/queryImage.jpg", cv2.IMREAD_GRAYSCALE)
if query_img is None:
    raise FileNotFoundError("Images/queryImage.jpg not found")

# Initialize ORB
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
kp_query, des_query = orb.detectAndCompute(query_img, None)

if des_query is None:
    raise ValueError("Could not detect features in query image.")
    # Optionally, handle this more gracefully if no features in query is possible

# Use BFMatcher for knnMatch
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Load video
video_path = "Images/video.MOV"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

start_time = time.time()

# Tracking state
tracking_roi = None  # Format: (x, y, w, h)
while True:
    ret, frame = cap.read()
    if not ret:
        break # End of video

    if frame is None:
         print("Warning: Received empty frame.")
         continue

    h, w = frame.shape[:2]

    # --- Decide crop region based on tracking state ---
    if tracking_roi is None:
        # Initial search or lost object, use larger margin or initial crop
        margin = DETECTION_MARGIN
        if 'last_roi' in locals() and last_roi is not None:
             # If we just lost it, use detection margin around last known spot
             x_last, y_last, w_last, h_last = last_roi
             x1 = max(x_last - margin, 0)
             y1 = max(y_last - margin, 0)
             x2 = min(x_last + w_last + margin, w)
             y2 = min(y_last + h_last + margin, h)
        else:
            # First frame or completely lost, use initial center crop
            ch = int(h * INIT_CROP_SCALE)
            cw = int(w * INIT_CROP_SCALE)
            x1 = (w - cw) // 2
            y1 = (h - ch) // 2
            x2 = x1 + cw
            y2 = y1 + ch
        last_roi = None # Clear last_roi once we search a wider area

    else:
        # Currently tracking, use smaller refinement margin around last known spot
        margin = REFINEMENT_MARGIN
        x_curr, y_curr, w_curr, h_curr = tracking_roi
        x1 = max(x_curr - margin, 0)
        y1 = max(y_curr - margin, 0)
        x2 = min(x_curr + w_curr + margin, w)
        y2 = min(y_curr + h_curr + margin, h)
        last_roi = tracking_roi # Store last good location in case we lose it

    # Ensure crop dimensions are positive
    cw = x2 - x1
    ch = y2 - y1

    if cw <= 0 or ch <= 0:
         # If crop is invalid, reset tracking and try wider search next frame
         print("Warning: Invalid crop size, resetting tracking.")
         tracking_roi = None
         continue

    # Perform processing on the cropped region
    cropped = frame[y1:y2, x1:x2]
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    kp_frame, des_frame = orb.detectAndCompute(gray_crop, None)

    # --- Feature Matching and Homography ---
    if des_frame is None or len(kp_frame) < MIN_MATCH_COUNT:
        # Not enough features in crop, reset tracking
        tracking_roi = None
        continue

    # Perform KNN matching
    matches = bf.knnMatch(des_query, des_frame, k=2)

    # Apply ratio test
    good_matches = []
    if matches: # Ensure matches list is not empty
        for m, n in matches:
            if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        # Extract points from good matches
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None and mask is not None and np.sum(mask) > MIN_MATCH_COUNT/2: # Check if M is valid and found reasonable inliers
            hq, wq = query_img.shape
            box = np.float32([[0, 0], [0, hq], [wq, hq], [wq, 0]]).reshape(-1, 1, 2)

            # Apply perspective transform to the box corners
            dst_corners = cv2.perspectiveTransform(box, M)

            # Offset back to full frame coordinates
            dst_corners_full = dst_corners + np.array([[[x1, y1]]])

            # Draw box on the full frame
            frame = cv2.polylines(frame, [np.int32(dst_corners_full)], True, (0, 255, 0), 3, cv2.LINE_AA)

            # Update tracking_roi for next frame using the bounding box of the transformed corners
            x, y, w_roi, h_roi = cv2.boundingRect(np.int32(dst_corners_full))
            tracking_roi = (x, y, (int)(w_roi*3/4), (int)(h_roi*3/4))
            # last_roi is kept as the current tracking_roi if successful

        else:
            # Homography estimation failed or too few inliers, reset tracking
            tracking_roi = None
            # last_roi retains the previous successful location for a wider search next frame
    else:
        # Not enough good matches found, reset tracking
        tracking_roi = None
        # last_roi retains the previous successful location for a wider search next frame


    # Optional: Show frame
    # cv2.imshow("Tracking", frame)
    # if cv2.waitKey(1) == 27: # Press 'ESC' to exit
    #     break

# Release resources
cap.release()
end_time = time.time()

cv2.destroyAllWindows()
print(f"Execution time for processing video: {end_time - start_time:.2f} seconds")