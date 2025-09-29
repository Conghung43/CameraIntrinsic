import cv2
import numpy as np

def is_frame_sharp(frame, threshold=100.0):
    """
    Determines if a video frame is sharp based on the variance of the Laplacian.

    Args:
        frame (numpy.ndarray): The video frame (in BGR format).
        threshold (float): Sharpness threshold. Higher value means stricter sharpness criteria.

    Returns:
        bool: True if the frame is sharp, False if blurry.
        float: The computed variance of the Laplacian (sharpness score).
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the Laplacian
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = laplacian.var()

    # Check if the variance exceeds the threshold
    is_sharp = variance > threshold

    return is_sharp, variance

def process_video(video_path, threshold=100.0):
    """
    Processes a video to analyze the sharpness of its frames.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Sharpness threshold.

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    frame_count = 0
    sharp_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        # Check if the current frame is sharp
        croppedFrame = frame[300:400,500:600]
        is_sharp, variance = is_frame_sharp(croppedFrame, threshold)
        if is_sharp:
            sharp_frames += 1
            cv2.imwrite("output.jpg", croppedFrame)
            print(f"Frame {frame_count}: Sharp (Variance: {variance:.2f})")
        else:
            print(f"Frame {frame_count}: Blurry (Variance: {variance:.2f})")

        # Optional: Display the frame (press 'q' to exit early)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total Frames: {frame_count}")
    print(f"Sharp Frames: {sharp_frames}")
    print(f"Blurry Frames: {frame_count - sharp_frames}")

# Example Usage
if __name__ == "__main__":
    video_path = "Images/2025-01-06 15.12.13.mp4"  # Replace with the path to your video
    sharpness_threshold = 1600.0  # Adjust the threshold as needed
    process_video(video_path, sharpness_threshold)