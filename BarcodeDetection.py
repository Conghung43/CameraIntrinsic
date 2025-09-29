import cv2
from pyzxing import BarCodeReader

# Load the image
image = cv2.imread('Images/img.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Otsu's Thresholding
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Edge Detection
# edges = cv2.Canny(thresh, 50, 150)

# Save the pre-processed image
cv2.imwrite('preprocessed_image.jpg', thresh)

# Decode using ZXing
reader = BarCodeReader()
barcodes = reader.decode_array(thresh)

# Print the results
print(barcodes)