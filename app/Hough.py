import cv2 as cv
import numpy as np
import math

def hough_line_transform(img, rho_resolution=1, theta_resolution= np.pi / 180, threshold= 150):
    # Check if image is loaded fine
    if img is None:
        print('Error: Image is None')
        return None

    dst = cv.Canny(img, 50, 200, None, 3)

    # Copy edges to the image that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    # Calculate the maximum possible rho value
    max_rho = int(math.sqrt(img.shape[0]**2 + img.shape[1]**2))

    # Initialize accumulator array
    accumulator = np.zeros((2 * max_rho, int(np.pi / theta_resolution)), dtype=np.uint32)

    # Perform Hough Transform
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if dst[y, x] != 0:
                for theta_index in range(accumulator.shape[1]):
                    theta = theta_index * theta_resolution
                    rho = int((x * math.cos(theta) + y * math.sin(theta)) / rho_resolution)    
                    accumulator[rho + max_rho, theta_index] += 1

    # Find lines with votes above threshold
    lines = []
    for rho_index in range(accumulator.shape[0]):
        for theta_index in range(accumulator.shape[1]):
            if accumulator[rho_index, theta_index] > threshold:
                rho = rho_index - max_rho
                theta = theta_index * theta_resolution
                lines.append((rho, theta))

    # Draw the detected lines
    for line in lines:
        rho, theta = line
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    return cdst

# Example usage
# img = cv.imread('your_image_path.jpg', cv.IMREAD_GRAYSCALE)
# hough_line_transform_manual(img, rho_resolution=1, theta_resolution=np.pi/180, threshold=150)
