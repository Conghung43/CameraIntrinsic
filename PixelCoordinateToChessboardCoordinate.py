import cv2
import numpy as np
import ChessboardTransformMatrix

def DrawAxis(img, corners_3d, rvecs, tvecs, camera_matrix, distortion_coeffs):
    # Draw origin (coordinate axes)
    for i in range(len(corners_3d)):
        axisLength = 10
        #origin = np.array([[i*axisLength, 0, 0], [axisLength + i*axisLength, 0, 0], [i*axisLength, axisLength, 0], [i*axisLength, 0, axisLength]], dtype=np.float32)  # Endpoints of axes in world coordinates
        origin = np.array([[corners_3d[i][0], corners_3d[i][1], corners_3d[i][2]], [corners_3d[i][0] + axisLength, corners_3d[i][1], corners_3d[i][2]], [corners_3d[i][0], corners_3d[i][1] + axisLength, corners_3d[i][2]], [corners_3d[i][0], corners_3d[i][1], corners_3d[i][2] + axisLength]], dtype=np.float32)
        axes_points, _ = cv2.projectPoints(origin, rvecs, tvecs, camera_matrix, distortion_coeffs)

        # Convert to tuple of integers
        axes_points = tuple(map(tuple, np.int32(axes_points).reshape(-1, 2)))

        # Draw coordinate axes
        img = cv2.line(img, axes_points[0], axes_points[1], (0, 0, 255), 3)  # X-axis (red)
        img = cv2.line(img, axes_points[0], axes_points[2], (0, 255, 0), 3)  # Y-axis (green)
        img = cv2.line(img, axes_points[0], axes_points[3], (255, 0, 0), 3)  # Z-axis (blue)

    # Show the image with coordinate axes
    cv2.imshow('Coordinate Axes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pixel2world(pixel_coords, camera_matrix, distortion_coeffs, rvecs, tvecs):
    # Convert pixel coordinates to normalized coordinates
    u, v = pixel_coords
    u_norm = (u - camera_matrix[0, 2]) / camera_matrix[0, 0]
    v_norm = (v - camera_matrix[1, 2]) / camera_matrix[1, 1]

    # Undistort normalized coordinates
    undist_coords = cv2.undistortPoints(np.array([[u_norm, v_norm]], dtype=np.float32), camera_matrix, distortion_coeffs)

    # SolvePnP to get world coordinates
    _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, camera_matrix, distortion_coeffs)
    R, _ = cv2.Rodrigues(rvecs)
    world_coords = np.dot(R.T, objp.T) + np.repeat(tvecs, 1, axis=1)

    return world_coords.squeeze()

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
    points_dst_list = [(point[0][0], point[0][1]) for point in points_dst]

    return points_dst_list

# Chessboard size
board_width = 9
board_height = 6

# Grid size in mm
grid_size_mm = 16.45

# Prepare object points
objp = np.zeros((board_width * board_height, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2) * grid_size_mm

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load camera intrinsic parameters (you need to provide this)
# Example: focal_length_mm = [fx, fy] and principal_point_mm = [cx, cy]
focal_length_mm = [800, 800]  # example values, replace with your camera's focal length
principal_point_mm = [320, 240]  # example values, replace with your camera's principal point

# Camera intrinsic matrix
camera_matrix = np.array([[focal_length_mm[0], 0, principal_point_mm[0]],
                           [0, focal_length_mm[1], principal_point_mm[1]],
                           [0, 0, 1]], dtype=np.float32)

# Load camera distortion coefficients (you need to provide this)
# Example: dist_coeffs = [k1, k2, p1, p2, k3]
dist_coeffs = np.zeros((5, 1))  # example values, replace with your camera's distortion coefficients

calibration_data = np.load('calibration_data_hannah.npz')  # Load calibration data obtained from cv2.calibrateCamera
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

# Load the image
image = cv2.imread('/Users/hungnguyencong/Downloads/rawImg.jpg')  # Replace 'chessboard_image.jpg' with your image path

# Undistorsion image
image = cv2.undistort(image, camera_matrix, dist_coeffs)
cv2.imwrite("transformed_image_undistor.jpg", image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# Find chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    # Refine corner locations for more accurate results
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # imgpoints.append(corners2)

    # Draw and display the corners
    # cv2.drawChessboardCorners(image, (board_width, board_height), corners2, ret)
    # cv2.imshow('Chessboard Corners', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Calibrate the camera (estimate camera matrix and distortion coefficients)
    #ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Calculate the rotation and translation vectors
    ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, camera_matrix, dist_coeffs)



    u, v = 942, 708  # Replace with your actual pixel coordinates

    # Get the 3D position of the point
    world_coords = pixel2world((u, v), camera_matrix, dist_coeffs, rvecs, tvecs)





    # Convert rotation vectors to rotation matrix
    R, _ = cv2.Rodrigues(rvecs)

    R_inv = np.linalg.inv(R)
    T_inv = -np.dot(R_inv, tvecs)
    cam2ChessTransformation_matrix = np.vstack((np.hstack((R_inv, T_inv)), [0, 0, 0, 1]))

    chessToCamTransformation_matrix = np.linalg.inv(cam2ChessTransformation_matrix)

    # Calculate 3D position of corners on chessboard plane
    corners_3d = np.dot(R.T, objp.T) + np.repeat(tvecs, board_width * board_height, axis=1)

    print("3D Positions of Chessboard Corners (in mm):")
    print(corners_3d)

    #point_2d = np.array([[756], 518]], dtype=np.float32)  # Example point coordinates, replace with your actual coordinates
#//retval, mask = cv2.findHomography(objp[:, :2], corners)

    

    # Define the coordinates of the corners in the image
    src_points = corners.reshape(-1, 2)
    corner_points = np.array([corners[0], corners[board_width - 1], corners[-1], corners[-board_width]], dtype=np.float32)
    

    # Define the coordinates of the corners in the desired output image
    dst_points = np.array([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, int(image.shape[1]*(board_height-1)/(board_width-1)) - 1], [0, int(image.shape[1]*(board_height-1)/(board_width-1)) - 1]], dtype=np.float32)

    # Calculate the homography
    H, _ = cv2.findHomography(corner_points, dst_points)

    # Save homography
    file_path = "h_matrix.txt"
    ChessboardTransformMatrix.save_matrix(file_path, H)

    points_src = [(3980, 2468), (4397, 845)]

    # Convert points from source image to destination image
    points_dst = convert_points_with_homography(points_src, H)

    # Convert point on destination image to 3D
    point_dst_3d = np.array(points_dst[0])/(dst_points[2])*np.array([grid_size_mm*(board_width-1), grid_size_mm*(board_height-1)])

    point_dst_3d = np.append(point_dst_3d, 0)
    point_dst_3d = np.append(point_dst_3d, 1)
    objectToRobotTransformMatrix = np.dot(ChessboardTransformMatrix.ChessboardToRobotMatrix(),point_dst_3d)






    # Apply the transformation
    transformed_image = cv2.warpPerspective(image, H, (image.shape[1], int(image.shape[1]*(board_height-1)/(board_width-1))))

    # Save the transformed image
    cv2.imwrite('transformed_image.jpg', transformed_image)
    
    DrawAxis(image, objp, rvecs, tvecs, camera_matrix, dist_coeffs)

else:
    print("Chessboard corners not found in the image.")
