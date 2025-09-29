import cv2
import numpy as np

# Configuration
pattern_count = 7
pattern_width = 1280
pattern_height = 720
camera_id = 0  # usually 0 or 1

# Calibration constants (example values, replace with your calibration)
focal_length = 800  # in pixels
baseline = 100 

# Generate stripe patterns (vertical stripes)
def generate_patterns():
    patterns = []
    for i in range(pattern_count):
        pattern = np.zeros((pattern_height, pattern_width), dtype=np.uint8)
        stripe_width = pattern_width // (2 ** (i + 1))
        for x in range(0, pattern_width, stripe_width * 2):
            pattern[:, x:x+stripe_width] = 255
        patterns.append(pattern)
    return patterns

# Initialize camera
cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, pattern_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pattern_height)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Generate patterns
patterns = generate_patterns()

# Project and capture
captured_images = []
for i, pattern in enumerate(patterns):
    # Show pattern fullscreen
    cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Projector", pattern)
    cv2.waitKey(50)  # Short initial wait for display to update
    
    # Wait for camera image to stabilize
    prev_frame = None
    stable_count = 0
    max_attempts = 10  # Maximum number of attempts to prevent infinite loop
    attempts = 0
    
    while stable_count < 2 and attempts < max_attempts:  # Need 3 consecutive stable frames
        ret, current_frame = cap.read()
        if not ret:
            print(f"Failed to read camera during stabilization for pattern {i+1}")
            break
            
        if prev_frame is not None:
            # Calculate difference between consecutive frames
            diff = cv2.absdiff(current_frame, prev_frame)
            mean_diff = np.mean(diff)
            
            # If difference is small enough, count as stable
            if mean_diff < 5.0:  # Threshold for stability (may need adjustment)
                stable_count += 1
            else:
                stable_count = 0  # Reset if unstable
                
        prev_frame = current_frame.copy()
        attempts += 1
        cv2.waitKey(33)  # ~30fps
    
    # Capture final image after stabilization
    ret, frame = cap.read()
    if ret:
        captured_images.append(frame)
        print(f"Captured image for pattern {i+1} after {attempts} frames")
    else:
        print(f"Failed to capture image for pattern {i+1}")

height, width, channels = captured_images[0].shape
codes = np.zeros((height, width, channels), dtype=np.uint16)

# Threshold to binary and build binary code per pixel
for i, img in enumerate(captured_images):
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    codes |= (binary.astype(np.uint16) << i)

# Calculate projector position (code) per pixel
projector_positions = (codes / (2**pattern_count - 1)) * (pattern_width - 1)

# Simulate disparity (difference between projector and camera positions)
# Here we assume camera x-coordinates are pixel indices
camera_x = np.tile(np.arange(width), (height, 1))
# Expand camera_x to match the shape of projector_positions (height, width, 3)
camera_x = np.repeat(camera_x[:, :, np.newaxis], 3, axis=2)
disparity = np.abs(projector_positions - camera_x)
disparity[disparity == 0] = 0.1  # avoid division by zero

# Depth calculation using stereo triangulation formula
depth = (focal_length * baseline) / disparity  # in mm

# Normalize and visualize depth map
depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_vis = depth_vis.astype(np.uint8)
color_map = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

cv2.imshow("Depth Map", color_map)
cv2.imwrite("depth_map.png", color_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Depth calculation complete!")