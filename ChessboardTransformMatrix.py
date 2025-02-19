import numpy as np

def save_matrix(file_path, matrix):
    np.savetxt(file_path, matrix)

def load_matrix(file_path):
    return np.loadtxt(file_path)

def ChessboardToRobotMatrix():
    # Define the corner points of the chessboard in the robot world coordinate system
    # chessboard_points = np.array([ [0.582, 6.766,  -30],
    #                               [82.896, 7.839,  -30],
    #                             [81.206, 139.571,  -30]])

    chessboard_points = np.array([
                                    # [6.6,5.3,-30],#[88.931, 7.095,  -30],
                                    # [3.347, 137.492, -30],
                                    # [86.034, 139.211,  -30]

                                    [4.398,7.555,-30],#[88.931, 7.095,  -30],
                                    [0.658, 139.407, -30],
                                    [82.803, 141.466,  -30]

                                ])

    # Define the origin corner point of the chessboard in the robot world coordinate system
    origin_point = chessboard_points[1]

    # Calculate the vectors of the chessboard coordinate system axes
    x_axis = chessboard_points[0] - origin_point
    y_axis = chessboard_points[2] - origin_point

    # Normalize the axes vectors
    x_axis = x_axis.astype(float) / np.linalg.norm(x_axis)
    y_axis = y_axis.astype(float) / np.linalg.norm(y_axis)

    # Calculate the z-axis (normal to the chessboard plane)
    z_axis = np.cross(x_axis, y_axis)

    # Construct the transformation matrix
    transform_matrix = np.array([ x_axis,y_axis, z_axis, origin_point]).T
    transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])  # Append the fourth row
    print("Transformation matrix from chessboard to robot world coordinate system:")
    print(transform_matrix)
    return transform_matrix

# ChessboardToRobotMatrix()
file_path = "transform_matrix_CR_v2.txt"
save_matrix(file_path, ChessboardToRobotMatrix())

loaded_matrix = load_matrix(file_path)
print("Loaded Matrix:")
print(loaded_matrix)
