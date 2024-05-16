import os
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces

# Load the Olivetti Faces dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces.images  # Original images

# Create a directory to save the images
os.makedirs('olivetti_faces', exist_ok=True)

# Save each image to a file
for i, image in enumerate(images):
    img = Image.fromarray((image * 255).astype(np.uint8))  # Convert to 8-bit grayscale
    filename = f'olivetti_faces/face_{i}.png'
    img.save(filename)
    print(f'Saved {filename}')