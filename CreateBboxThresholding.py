import cv2
import numpy as np

# Load the image
image = cv2.imread("/Users/nguyenconghung/Desktop/2025-02-14 13.41.45.jpg")  # Change to your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray.jpg', gray)
# Reshape image into a 1D array (each pixel is a sample)
pixels = image.reshape(-1, 1).astype(np.float32)

# Apply K-Means clustering with 2 clusters (foreground & background)
K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Compute the threshold as the midpoint of the two cluster centers
threshold_value = np.mean(centers)
print(threshold_value)
# Apply thresholding
_, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding box if a contour is found
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)  # Get bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle

# Show the result
cv2.imshow("Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
