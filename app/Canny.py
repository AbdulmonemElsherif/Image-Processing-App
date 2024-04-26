import cv2
import numpy as np


def to_grey(img: np.ndarray):
    if len(img.shape) == 3:
        [r_coef, g_coef, b_coef] = [0.3, 0.59, 0.11]
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        grey_img = r_coef * r + g_coef * g + b_coef * b
    else:
        grey_img = img  
    return grey_img


def generate_gaussian_filter(sigma: float, filterSize: int):
 gaussianFilter = np.zeros((filterSize,filterSize))
 center = filterSize//2 
 #Create the actual filter
 for x in range(filterSize):
  for y in range(filterSize):
   gaussianFilter[x,y] = (1/(2.0 * np.pi * sigma**2.0))*np.exp(-((x-center)**2.0+(y-center)**2.0)/(2.0*sigma**2.0))
 return gaussianFilter
    
def convolution(img: np.ndarray, kernel: np.ndarray):
    convoluted = np.zeros(img.shape)
    kernelSize = kernel.shape[0]
    halfKernel = kernelSize // 2

    # Handle cases where img.shape is a single value
    if len(img.shape) == 1:
        width = height = img.shape[0]
    else:
        [width, height] = img.shape

    # Pad image manually with duplicates of the edge pixels
    padded_img = np.zeros((width + 2 * halfKernel, height + 2 * halfKernel))
    padded_img[halfKernel:halfKernel + width, halfKernel:halfKernel + height] = img

    for x in range(width):
        for y in range(height):
            img_portion = padded_img[x:x + kernelSize, y:y + kernelSize]
            convoluted[x, y] = np.sum(img_portion * kernel)
    return convoluted


    
def sobel_edge_detection(blurredImg: np.ndarray):
 matrix_x = np.array(
     [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]], np.float32
 )
 matrix_y = np.array(
         [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]], np.float32
     )
 gx = convolution(blurredImg, matrix_x)
 gy = convolution(blurredImg, matrix_y)
 g = np.sqrt(gx**2.0 + gy**2.0)
 theta = np.arctan2(gy, gx)
 return g, theta, gx, gy
    
def non_max_suppression(g: np.ndarray, theta: np.ndarray):
 m, n = g.shape
 angle = np.degrees(theta)
 
 for x in range(1, m - 1):
  for y in range(1, n - 1):
   # Ensure angle is non-negative
   if angle[x, y] < 0:
    angle[x, y] += 180
   
   # Determine neighboring pixels based on gradient direction
   if 0 <= angle[x, y] < 22.5:
    neigbourPixel1, neigbourPixel2 = g[x, y-1], g[x, y+1]
   elif 22.5 <= angle[x, y] < 67.5:
    neigbourPixel1, neigbourPixel2 = g[x-1, y+1], g[x+1, y-1]
   elif 67.5 <= angle[x, y] < 112.5:
    neigbourPixel1, neigbourPixel2 = g[x-1, y], g[x+1, y]
   elif 112.5 <= angle[x, y] < 157.5:  
    neigbourPixel1, neigbourPixel2 = g[x+1, y+1], g[x-1, y-1]
   else:
    neigbourPixel1, neigbourPixel2 = g[x, y-1], g[x, y+1]

   if g[x, y] < neigbourPixel1 or g[x, y] < neigbourPixel2:
    g[x, y] = 0 
 return g

def double_threshold(img: np.ndarray, highThresh: float, lowThresh: float):
 [n,m] = img.shape

 marking = np.zeros(img.shape)
 for x in range(n):
  for y in range(m):
   if img[x,y] > highThresh:
    marking[x,y] = 255
   elif img[x,y] < lowThresh:
    marking[x,y] = 0
   else:
    marking[x,y] = 25
 return marking

def hysteresis(img: np.ndarray):
 [n,m] = img.shape

 for x in range(n-1):
  for y in range(m-1):
   if img[x,y] == 25:
    if ((img[x+1,y] == 255) or (img[x-1,y] == 255) or (img[x+1,y+1] == 255) or (img[x+1,y-1] == 255) or
    (img[x-1,y+1] == 255) or (img[x-1,y-1] == 255) or (img[x,y+1] == 255) or (img[x,y-1] == 255)):
     img[x,y] = 255
    else:
     img[x,y] = 0
 return img

def canny_edge_detector(img, sigma, ksize, lowthreshold, highthreshold):
    grey = to_grey(img)
    kernel = generate_gaussian_filter(sigma, ksize)
    blurred = convolution(grey, kernel) 

    [g, theta, gx, gy] = sobel_edge_detection(blurred)

    suppressedImg = non_max_suppression(g, theta)

    marks = double_threshold(suppressedImg, lowthreshold, highthreshold)
    final_image = hysteresis(marks)
        
    return final_image
