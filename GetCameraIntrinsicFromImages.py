import numpy as np
import cv2
import os

# Define the size of the chessboard and grid size
pattern_size = (9, 6)  # Number of inner corners along the rows and columns of the chessboard
square_size = 20  # Size of each square in mm

# Prepare object points, like (0,0,0), (20,0,0), (40,0,0) ....,(160,100,0)
object_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images
object_points_list = []  # 3D points in real world space
image_points_list = []   # 2D points in image plane

# Capture images from camera
num_images = 50  # Number of images to capture
image_count = 0

def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp','.JPG')):  # Add more extensions if needed
            image_path = os.path.join(folder_path, filename)
            
            #if image is not None:
            images.append(image_path)
    return images

images = read_images_from_folder("/Users/hungnguyencong/Downloads/IphoneChessboard2")

# for frame in images:
# Capture images from camera
num_images = 200  # Number of images to capture
image_count = 0

#cap = cv2.VideoCapture("images/V_20240314_111248_N0.mp4")

# while image_count < num_images:
#     ret, frame = cap.read()
#     if not ret:
#         break
for path in images:
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points
    if ret:
        print(path)
        object_points_list.append(object_points)
        image_points_list.append(corners)

        # Draw and display the corners
        # cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        # cv2.imshow('Chessboard', frame)
        # cv2.waitKey(500)  # Adjust wait time as needed

        image_count += 1
        print(image_count)

#cap.release()
cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
    object_points_list, image_points_list, gray.shape[::-1], None, None)

np.savez('calibration_data_iphone.npz', mtx=camera_matrix, dist=distortion_coefficients)

# Print the intrinsic parameters
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(distortion_coefficients)

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print("Focal Length (fx, fy):", fx, fy)
print("Principal Point (cx, cy):", cx, cy)