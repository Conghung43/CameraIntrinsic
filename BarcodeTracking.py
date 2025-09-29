import cv2
import numpy as np
# from pyzbar.pyzbar import decode

# Initialize video capture
video_path = "Images/IMG_8606.MOV"
cap = cv2.VideoCapture(video_path)

# Initialize tracker
trackers = cv2.legacy.MultiTracker_create()

# Frame counter
frame_count = 0

# Function to detect barcodes and return bounding boxes
def detect_barcodes(frame):
    barcodes = decode(frame)
    bboxes = []
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        bboxes.append((x, y, w, h))
    return bboxes

# Function to re-track using optical flow
def retrack_with_optical_flow(prev_frame, curr_frame, prev_bboxes):
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    new_bboxes = []
    for bbox in prev_bboxes:
        x, y, w, h = bbox
        roi_prev = gray_prev[y:y+h, x:x+w]

        # Detect features in the previous bounding box
        p0 = cv2.goodFeaturesToTrack(roi_prev, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if p0 is not None:
            p0[:, 0, 0] += x
            p0[:, 0, 1] += y

            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, p0, None)

            # Filter good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0:
                x_new, y_new, w_new, h_new = cv2.boundingRect(good_new)
                new_bboxes.append((x_new, y_new, w_new, h_new))
            else:
                new_bboxes.append(bbox)  # Fallback to previous bbox if no good points
        else:
            new_bboxes.append(bbox)  # Fallback to previous bbox if no features detected

    return new_bboxes

# Function to refine bounding boxes using homography and feature matching
def refine_bboxes_with_homography(prev_frame, curr_frame, prev_bboxes):
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        new_bboxes = []
        for bbox in prev_bboxes:
            x, y, w, h = bbox
            corners = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, matrix)
            x_new, y_new, w_new, h_new = cv2.boundingRect(transformed_corners)
            new_bboxes.append((x_new, y_new, w_new, h_new))

        return new_bboxes
    else:
        return prev_bboxes  # Fallback to previous bboxes if not enough matches

# Process video frames
prev_frame = None
prev_bboxes = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detect barcodes in frame 10
    if frame_count == 10:
        bboxes = [(637, 483, 986 - 637, 656 - 483)]  # detect_barcodes(frame)
        prev_bboxes = bboxes
        for bbox in bboxes:
            tracker = cv2.legacy.TrackerCSRT_create()
            trackers.add(tracker, frame, tuple(bbox))

    # Update trackers for subsequent frames
    elif frame_count > 10:
        # Use optical flow or feature matching to track barcodes
        if prev_frame is not None:
            boxes = retrack_with_optical_flow(prev_frame, frame, prev_bboxes)

        # Refine bounding boxes using homography and feature matching
        if prev_frame is not None:
            boxes = refine_bboxes_with_homography(prev_frame, frame, boxes)

        # Draw bounding boxes
        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        prev_bboxes = boxes

    # Store the current frame as the previous frame
    prev_frame = frame.copy()

    # Display the frame
    cv2.imshow('Barcode Tracking', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()