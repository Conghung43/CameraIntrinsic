import cv2
import numpy as np

def convert_image(image_A, K_A, K_B, T_BA):
    # Undistort image A using camera matrix K_A
    undistorted_A = cv2.undistort(image_A, K_A, None)

    # Get image size
    height, width = undistorted_A.shape[:2]

    # Generate new camera matrices for rectification
    R_BA = np.linalg.inv(T_BA[:3, :3])  # Rotation from camera A to camera B
    new_K_A, _ = cv2.getOptimalNewCameraMatrix(K_A, None, (width, height), 1, (width, height))
    new_K_B, _ = cv2.getOptimalNewCameraMatrix(K_B, None, (width, height), 1, (width, height))

    # Rectify images
    mapx_A, mapy_A = cv2.initUndistortRectifyMap(K_A, None, R_BA, new_K_A, (width, height), cv2.CV_32FC1)
    rectified_A = cv2.remap(undistorted_A, mapx_A, mapy_A, cv2.INTER_LINEAR)

    mapx_B, mapy_B = cv2.initUndistortRectifyMap(K_B, None, np.eye(3), new_K_B, (width, height), cv2.CV_32FC1)
    rectified_B = cv2.remap(image_A, mapx_B, mapy_B, cv2.INTER_LINEAR)

    return rectified_A, rectified_B

# Example usage
image_A = cv2.imread("image_from_camera_A.jpg")  # Load image from camera A
K_A = np.array([[fx_A, 0, cx_A], [0, fy_A, cy_A], [0, 0, 1]])  # Intrinsic matrix for camera A
K_B = np.array([[fx_B, 0, cx_B], [0, fy_B, cy_B], [0, 0, 1]])  # Intrinsic matrix for camera B
T_BA = np.array([[r11, r12, r13, t_x], [r21, r22, r23, t_y], [r31, r32, r33, t_z]])  # Relative transformation from camera B to camera A

rectified_image_A, rectified_image_B = convert_image(image_A, K_A, K_B, T_BA)

# Display or save rectified images
cv2.imshow("Rectified Image A", rectified_image_A)
cv2.imshow("Rectified Image B", rectified_image_B)
cv2.waitKey(0)
cv2.destroyAllWindows()
