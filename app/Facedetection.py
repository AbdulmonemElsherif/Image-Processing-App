import numpy as np
import cv2
import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load the face images from a directory
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img.flatten())
            try:
                labels.append(int(filename.split('_')[1].split('.')[0]))  # Assuming filenames have labels after underscore
            except ValueError:
                print(f"Filename {filename} does not contain an integer after underscore. Skipping this file.")
    return images, labels

# Load the dataset
images, labels = load_images_from_folder('olivetti_faces')
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Compute the PCA
n_components = 150
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
mean_face = pca.mean_
eigenfaces = pca.components_
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Function to find the nearest neighbor using dot product
def find_nearest_neighbor(training_data, training_labels, test_image):
    dot_products = np.dot(training_data, test_image)
    nearest_neighbor_index = np.argmax(dot_products)
    return nearest_neighbor_index, training_labels[nearest_neighbor_index]

# Function to get the image corresponding to a given label
def get_image_by_label(images, labels, label):
    for img, lbl in zip(images, labels):
        if lbl == label:
            return img.reshape((64, 64))  # Adjust based on your image shape
    return None

# GUI implementation
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition using Eigenfaces")

        self.label = Label(root, text="Choose an image to recognize:")
        self.label.pack()

        self.upload_btn = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.upload_canvas = Canvas(root, width=300, height=300)
        self.upload_canvas.pack(side=LEFT, padx=10, pady=10)

        self.recognized_canvas = Canvas(root, width=300, height=300)
        self.recognized_canvas.pack(side=RIGHT, padx=10, pady=10)

        self.result_label = Label(root, text="")
        self.result_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            original_img = img.copy()
            img = img.flatten()
            img_pca = pca.transform([img])[0]

            nearest_neighbor_index, predicted_label = find_nearest_neighbor(X_train_pca, y_train, img_pca)
            recognized_img = get_image_by_label(X_train, y_train, predicted_label)
            
            self.show_image(self.upload_canvas, original_img, "Uploaded Image")
            if recognized_img is not None:
                self.show_image(self.recognized_canvas, recognized_img, "Recognized Image")
            self.result_label.config(text=f"Predicted Label: {predicted_label}")

    def show_image(self, canvas, img, title):
        img = cv2.resize(img, (300, 300))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=NW, image=img)
        canvas.image = img
        canvas.create_text(150, 280, text=title, fill="white")

# Run the application
root = Tk()
app = FaceRecognitionApp(root)
root.mainloop()
