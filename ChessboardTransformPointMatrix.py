import numpy as np

# Define the coordinates of the point
point = np.array([10, 20, 0, 1])  # Adding 1 to represent the point in homogeneous coordinates

# Define the translation vector to represent the position of the point
translation = np.array([0, 0, 0])

# Define the rotation matrix to represent the orientation of the point (identity matrix for no rotation)
rotation = np.eye(3)

# Create the homogeneous transformation matrix
pose_matrix = np.eye(4)
pose_matrix[:3, 3] = translation
pose_matrix[:3, :3] = rotation

# Apply the transformation to the point to get its pose
pose = np.dot(pose_matrix, point)

print("Pose of the point (10, 20, 0) relative to the origin:")
print(pose)
