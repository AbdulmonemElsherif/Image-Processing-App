import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces

# Load the Olivetti Faces dataset (or use your own dataset)
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data  # Flattened face images
n_samples, n_features = X.shape

# Compute the mean face
mean_face = np.mean(X, axis=0)

# Center the data by subtracting the mean face
X_centered = X - mean_face

# Perform PCA to find eigenvectors and eigenvalues
n_components = 100  # Choose the number of components (Eigenfaces)
pca = PCA(n_components=n_components)
eigenfaces = pca.fit_transform(X_centered)

# Reconstruct a face using k components
k_values = [5, 10, 30]  # Example: 5, 10, 30 components
for k in k_values:
    pca_k = PCA(n_components=k)
    X_k = pca_k.fit_transform(X_centered)
    reconstructed_face = pca_k.inverse_transform(X_k)
    reconstructed_face = mean_face + reconstructed_face[0]
    plt.figure()
    plt.imshow(reconstructed_face.reshape(faces.images[0].shape), cmap='gray')
    plt.title(f'Reconstructed Face (k={k})')
    plt.axis('off')

# Plot cumulative explained variance ratio
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Components')
plt.grid()

plt.show()
