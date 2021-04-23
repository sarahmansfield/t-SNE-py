import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_iris
import scipy.io
from tsne.tsne import TSNE, plot_tsne


def reduce_pca(X, ndim=30):
    """
    Uses PCA to reduce dimensionality of the data
    :param X: a high-dimensional matrix
    :param ndim: the number of dimensions to reduce X to
    :return: a lower dimensional matrix with ndim columns
    """
    standardized_data = StandardScaler().fit_transform(X)
    pca = PCA(n_components=ndim)
    principal_components = pca.fit_transform(standardized_data)
    return principal_components


# MNIST dataset
mnist = np.loadtxt("data/mnist2500_X.txt")
mnist_labels = np.loadtxt("data/mnist2500_labels.txt")
mnist_reduced = reduce_pca(mnist)
mnist_Y = TSNE(mnist_reduced)
plot_tsne(mnist_Y, mnist_labels)

# Olivetti faces dataset
olivetti = fetch_olivetti_faces()
olivetti_reduced = reduce_pca(olivetti.data)
olivetti_Y = TSNE(olivetti_reduced)
plot_tsne(olivetti_Y, olivetti.target)

# COIL20 dataset
mat = scipy.io.loadmat("COIL20.mat")
coil20 = mat['X']
coil20_labels = mat['Y'][:, 0]
coil20_reduced = reduce_pca(coil20)
coil20_Y = TSNE(coil20_reduced)
plot_tsne(coil20_Y, coil20_labels)

# Iris dataset
iris = load_iris()
iris_Y = TSNE(iris.data)
plot_tsne(iris_Y, iris.target)
