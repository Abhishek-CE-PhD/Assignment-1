import numpy as np
import matplotlib.pyplot as plt

def pca(X, n_components):
    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
    sorted_idx = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorted_idx]
    eigen_vecs = eigen_vecs[:, sorted_idx]
    eigen_vecs = eigen_vecs[:, :n_components]
    X_reduced = np.dot(X_meaned, eigen_vecs)
    return X_reduced, eigen_vals[:n_components], eigen_vecs

# example dataset
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

X_reduced, eigen_vals, eigen_vecs = pca(X, 2)

plt.scatter(X[:, 0], X[:, 1], label="Original Data")
origin = np.mean(X, axis=0)
for length, vector in zip(eigen_vals, eigen_vecs.T):
    v = vector * 2 * np.sqrt(length)
    plt.quiver(*origin, *v, angles="xy", scale_units="xy", scale=1, color="r")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.title("PCA with Eigenvectors")
plt.show()
