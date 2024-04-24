import numpy as np
import cv2

def HoughTransform(img, threshold, rho_resolution, theta_resolution, num_peaks):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.5)

    # Apply Canny edge detection
    edges = cv2.Canny(img_blur, threshold, threshold * 2)

    # Get edge coordinates
    y_idxs, x_idxs = np.nonzero(edges)

    # Calculate image diagonal
    img_diagonal = np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2))

    # Create an array to accumulate values
    accumulator = np.zeros((int(np.ceil(img_diagonal/rho_resolution)), int(np.ceil(180/theta_resolution))))

    # Compute the Hough Transform
    for i in range(len(x_idxs)):
        for theta in np.arange(0, 180, theta_resolution):
            rho = x_idxs[i]*np.cos(np.deg2rad(theta)) + y_idxs[i]*np.sin(np.deg2rad(theta))
            accumulator[int(np.ceil(rho/rho_resolution)), int(np.ceil(theta/theta_resolution))] += 1

    # Find the 'num_peaks' strongest lines
    peaks = np.argsort(accumulator.ravel())[-num_peaks:]
    rho_values = (peaks // accumulator.shape[1]) * rho_resolution
    theta_values = (peaks % accumulator.shape[1]) * theta_resolution

    return rho_values, theta_values