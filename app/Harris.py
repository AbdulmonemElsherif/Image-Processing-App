import numpy as np
import cv2

def to_grayscale(image):
    if len(image.shape) == 3:  
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image  
def compute_gradients(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobelx, sobely
def compute_structure_tensor(sobelx, sobely, window_size=3):
    Ixx = sobelx**2
    Ixy = sobelx*sobely
    Iyy = sobely**2
    

    kernel = cv2.getGaussianKernel(window_size, -1)
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    
    return Sxx, Sxy, Syy
def compute_harris_response(Sxx, Sxy, Syy, k=0.04):

    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy

    R = det - k * (trace**2)
    return R
def non_maximum_suppression(R, threshold=0.01):

    R[R < threshold * R.max()] = 0
    
    corners = np.nonzero(R)
    values = R[corners]
    
    # Sort by corner response
    idx = np.argsort(-values)
    corners = (corners[0][idx], corners[1][idx])
    
    
    return corners
def harris_corner_detector(image, window_size=3, k=0.04, threshold=0.01):
    gray_image = to_grayscale(image)
    sobelx, sobely = compute_gradients(gray_image)
    Sxx, Sxy, Syy = compute_structure_tensor(sobelx, sobely, window_size)
    R = compute_harris_response(Sxx, Sxy, Syy, k)
    corners = non_maximum_suppression(R, threshold)
    return corners
