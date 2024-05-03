import numpy as np 
import cv2
board_width = 9
board_height = 6
grid_size_mm = 19.15
def load_matrix(file_path):
    return np.loadtxt(file_path)

def convert_points_with_homography(points, H):
    """
    Convert a list of points from the source image to the destination image using a homography matrix.

    Args:
    - points: List of points in the source image, each point should be a tuple (x, y).
    - H: Homography matrix.

    Returns:
    - List of points in the destination image, each point is a tuple (x, y).
    """
    # Convert points to numpy array for easier manipulation
    points_np = np.array(points, dtype=np.float32)

    # Convert points from source image to destination image using perspective transform
    points_dst = cv2.perspectiveTransform(points_np.reshape(-1, 1, 2), H)

    # Convert points back to list of tuples
    points_dst_list = [np.array((point[0][0], point[0][1])) for point in points_dst]

    return points_dst_list

H_matrix = load_matrix('h_matrix.txt')
BR_matrix = load_matrix('transform_matrix_self.txt')

calibration_data = np.load('CameraIntrinsic/calibration_data_self.npz')  # Load calibration data obtained from cv2.calibrateCamera
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

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
    # continue
    cv2.imwrite("img.jpg", frame)
    # frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    

    # objectPose = 517,197

    # Ratio to pixel
    objectPose = np.array((458.5,384.6))#*np.array((frame.shape[1], frame.shape[0]))

    # Calibration
    # Convert points from source image to destination image
    points_dst = convert_points_with_homography(objectPose, H_matrix)

    # Convert point on destination image to 3D
    point_dst_3d = np.array(points_dst)/np.array((frame.shape[1],1200))*np.array([grid_size_mm*(board_width-1), grid_size_mm*(board_height-1)])

    # point_dst_3d = np.append(point_dst_3d, 0)
    # point_dst_3d = np.append(point_dst_3d, 1)
    # Array to append to each row
    to_append = np.array([0, 1])
    # Add to_append to each row
    result = np.hstack((point_dst_3d, np.tile(to_append, (point_dst_3d.shape[0], 1))))
    objectPose = [np.dot(BR_matrix, vector) for vector in result]
    print(objectPose)