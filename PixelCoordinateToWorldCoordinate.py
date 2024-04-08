import numpy as np
import cv2
import os 

# Define chessboard parameters
chessboard_size = (9, 6)  # Size of chessboard corners
square_size_mm = 20       # Size of each square in mm

# Define world coordinates of chessboard corners
world_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
world_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size_mm

# Load calibration data
calibration_data = np.load('calibration_data_iphone1.npz')  # Load calibration data obtained from cv2.calibrateCamera
camera_matrix = calibration_data['mtx']
distortion_coeffs = calibration_data['dist']


def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp','.JPG')):  # Add more extensions if needed
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
    return images

images = read_images_from_folder("/Users/hungnguyencong/Downloads/iphone_test")

for img in images:

    # Load an example image
    #img = cv2.imread('images/VivoCpuImage/1_5.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Refine corner locations for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Estimate pose of chessboard (rotation and translation vectors)
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(world_points, corners, camera_matrix, distortion_coeffs)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvecs)

        # Construct transformation matrix
        transformation_matrix = np.hstack((rotation_matrix, tvecs))

        print("Transformation matrix from chessboard to camera:")
        print(transformation_matrix)

        # Construct transformation matrix for the specific point
        # specific_point = corners[0]
        # specific_point_homogeneous = np.hstack((specific_point.reshape(3, 1), np.ones((1, 1))))
        # transformed_point = np.dot(transformation_matrix, specific_point_homogeneous)

        # print("Transformation matrix for the specific point:")
        # print(transformed_point)


        # Draw origin (coordinate axes)
        axisLength = 50
        origin = np.array([[0, 0, 0], [axisLength, 0, 0], [0, axisLength, 0], [0, 0, axisLength]], dtype=np.float32)  # Endpoints of axes in world coordinates
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

    else:
        print("Chessboard corners not found.")

