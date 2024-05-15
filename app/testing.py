import numpy as np
from skimage import io
from skimage import color
from skimage.transform import hough_ellipse
from skimage.feature import canny
from skimage.draw import ellipse_perimeter
from skimage.util import img_as_ubyte

def hough_ellipse_transform(img_path, min_size=100, max_size=120, accuracy=50, threshold=230, sigma=2.0, low_threshold=0.55, high_threshold=0.8):
    # Load picture, convert to grayscale, and detect edges
    if isinstance(img_path, str):
        image_rgb = io.imread(img_path)
    else:
        image_rgb = img_path

    if len(image_rgb.shape) == 3:
        image_gray = color.rgb2gray(image_rgb)
    else:
        image_gray = image_rgb

    edges = canny(image_gray, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

    # Perform the Hough Transform for ellipses
    result = hough_ellipse(edges, accuracy=accuracy, threshold=threshold, min_size=min_size, max_size=max_size)

    # Check if any ellipses were found
    if result.size > 0:
        result.sort(order='accumulator')

        # Extract ellipse parameters
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        image_rgb[cy, cx] = (0, 0, 255)  # Ensure these coordinates are within the image bounds
    else:
        print("No ellipses found")
    return image_rgb

# Example usage
image_path = 'Pictures/coffee.jpg'
result_image = hough_ellipse_transform(image_path)
io.imshow(result_image)
io.show()
