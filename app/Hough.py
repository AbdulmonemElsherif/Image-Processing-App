import cv2 as cv
import numpy as np
import math

from .Canny import canny_edge_detector

def hough_line_transform(img, theta_resolution=np.pi / 180, threshold=150, lowerThreshold=50, upperThreshold=200, ksize=3, sigma = 3):
    if img is None:
        print('Error: Image is None')
        return None

    edges = canny_edge_detector(img, sigma, ksize ,lowerThreshold, upperThreshold)

    img_height, img_width = img.shape[:2]
  
    diag_len = int(math.sqrt(np.square(img_height) + np.square(img_width)))

    accumulator = np.zeros((2 * diag_len, int(np.pi / theta_resolution)), dtype=np.uint32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if edges[y, x] != 0:  # If pixel is part of an edge
                for theta_index in range(accumulator.shape[1]):
                    theta = theta_index * theta_resolution
                    rho = int((x * math.cos(theta) + y * math.sin(theta)))
                    accumulator[rho + diag_len, theta_index] += 1

    lines = []
    for rho_index in range(accumulator.shape[0]):
        for theta_index in range(accumulator.shape[1]):
            if accumulator[rho_index, theta_index] > threshold:
                rho = rho_index - diag_len
                theta = theta_index * theta_resolution
                lines.append((rho, theta))
    edges_uint8 = cv.convertScaleAbs(edges)
    result = cv.cvtColor(edges_uint8, cv.COLOR_GRAY2BGR)
    for line in lines:
        rho, theta = line
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(result, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    return result

