import cv2
import numpy as np
from sklearn.cluster import KMeans

video_path = "output.avi"  # Your input video file
cap = cv2.VideoCapture(video_path)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

frame_rate = cap.get(cv2.CAP_PROP_FPS)
prev_center = None
raw_speeds = []
smoothed_speed = None
BUFFER_SIZE = 400

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 500]

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        center = (x + w // 2, y + h // 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        if prev_center is not None:
            px, py, pw, ph = cv2.boundingRect(prev_center)
            pCenter = (px + pw // 2, py + ph // 2)
            if (center[0] >  px and center[0] < px+ pw):
                dx = center[0] - pCenter[0]
                dy = center[1] - pCenter[1]
                distance = (dx ** 2 + dy ** 2) ** 0.5
                speed = distance * frame_rate
                raw_speeds.append(speed)
            else:
                # debugging
                print(f"Skipping speed calculation")
        # deep copy the center for next iteration
        prev_center = largest.copy()

    # Cluster every 300 speed readings
    if len(raw_speeds) >= BUFFER_SIZE:
        # speeds_np = np.array(raw_speeds[-BUFFER_SIZE:]).reshape(-1, 1)

        # # KMeans clustering into 2 groups
        # kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        # kmeans.fit(speeds_np)

        # # Get the cluster centers of the fist cluster group
        # cluster_centers = kmeans.cluster_centers_[0].flatten()
        # stable_speed = np.mean(cluster_centers)
        # smoothed_speed = stable_speed
        # raw_speeds.pop(0)# = []  # reset buffer
        smoothed_speed = np.average(raw_speeds)
        raw_speeds.pop(0)

    # Display the smoothed speed
    if smoothed_speed is not None:
        cv2.putText(frame, f"Stable Speed: {smoothed_speed:.2f} px/s", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Conveyor Belt Speed (Clustered)", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()