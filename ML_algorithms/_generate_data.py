import numpy as np
import matplotlib.pyplot as plt

# Generate html image
plt.savefig('tmp.png', bbox_inches='tight')
import base64
data_uri = base64.b64encode(open('tmp.png', 'rb').read()).decode('utf-8').replace('\n', '')
img_tag = '<img height="300" src="data:image/png;base64,%s">' % data_uri
with open("tmp.txt", "w") as text_file:
    text_file.write(img_tag)

###
# Test set from 
# archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
# Features are # of pixels in 8x8 grid
###

test = np.loadtxt("data/optdigits_test.txt", delimiter = ",")
X = test[:, 0:64]
y = test[:, 64]

# Plot an image
image = X[4].reshape((8, 8))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
imgplot = ax.imshow(image, cmap = "Greys", vmin = 0, vmax = 16)
plt.show()


###
# Generate normal classification
###
n_samples = 1000
n_features = 2
centers = [-1, 1]
E = np.random.uniform(size = (n_features, n_features))
X = np.concatenate([centers[0] + np.random.normal(size = (n_samples, n_features)) @ E, centers[1] + np.random.normal(size = (n_samples, n_features)) @ E])
Y = np.concatenate([np.repeat(1, n_samples), np.repeat(2, n_samples)])
M = np.column_stack([X, Y])
np.random.shuffle(M)
X = M[:, [0, 1]]
Y = M[:, 2]

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c = Y, s=1)
plt.show()

from sklearn.datasets import make_classification
plt.figure(figsize=(8, 8))
X, Y = make_classification(n_samples = n_samples, n_features=n_features, n_redundant=0, n_informative=n_features, n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=1)
plt.show()

###
# Generate 2 normal clusters
###
n_samples = 10
n_features = 2
centers = [-5, 5]
X = np.concatenate([centers[0] + np.random.normal(size = (n_samples, n_features)), centers[1] + np.random.normal(size = (n_samples, n_features))])
Y = np.concatenate([np.repeat(1, n_samples), np.repeat(2, n_samples)])
M = np.column_stack([X, Y])
np.random.shuffle(M)
X = M[:, [0, 1]]
Y = M[:, 2]

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c = Y, s=1)
plt.show()

from sklearn.datasets import make_blobs
plt.figure(figsize=(8, 8))
X, Y = make_blobs(n_samples = n_samples, n_features = n_features, centers=3)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=1)
plt.show()
