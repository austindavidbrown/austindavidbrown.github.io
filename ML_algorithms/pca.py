import numpy as np
import matplotlib.pyplot as plt

###
# Load the UCI optical digits test set
# preprocessed dataset is an 8x8 grid of # of pixels from 0-16 with labels
###
test = np.loadtxt("data/optdigits_test.txt", delimiter = ",")
X = test[:, 0:64]
y = test[:, 64]

# Plot digits
fig = plt.figure(figsize=(8, 6))
fig.tight_layout()
for i in range(0, 20):
    ax = fig.add_subplot(5, 5, i + 1)
    ax.imshow(X[i].reshape((8,8)), cmap = "Greys", vmin = 0, vmax = 16)
plt.show()


# Standardize with respect to the features not the rows
# Standardize across the rows!!!
mu = np.mean(X, axis = 0)
sigma = np.std(X, axis = 0)
sX = 1/(sigma + 10**(-10)) * (X - mu)

# The covariance matrix of the features! not the observations
E = 1/(sX.shape[0] - 1) * sX.T @ sX
spectrum, U = np.linalg.eig(E)
print("Eigenvectors:", spectrum)
print("Eigenvalues:", U)

spectrum_corr = spectrum / np.sum(spectrum)
print("Explained variability:", spectrum_corr)

K = 2
I = np.argsort(-spectrum)[0:K]
Z = sX @ U[:, I]

# Visualize in 2d
plt.style.use("default")
fig = plt.figure(figsize=(8, 6))
plt.tight_layout()
plt.scatter(Z[:, 0], Z[:, 1], s = 5, c = y)
plt.xlabel('Z1')
plt.ylabel('Z2')
plt.colorbar()
plt.show()

# Train test split
n_samples = X.shape[0]
n_TRAIN = int(.75 * n_samples)
I = np.arange(0, n_samples)
TRAIN = np.random.choice(I, n_TRAIN, replace = False)
TEST = np.setdiff1d(I, TRAIN)
X_train = X[TRAIN, :]
y_train = y[TRAIN]
X_test = X[TEST, :]
y_test = y[TEST]


# Using all of the features
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "lbfgs", max_iter = 1000, multi_class = "multinomial")
lr.fit(X_train, y_train)
print("Train accuracy:", 1/X_train.shape[0] * np.sum(lr.predict(X_train) == y_train))
print("Test accuracy", 1/X_test.shape[0] * np.sum(lr.predict(X_test) == y_test))


# Using PCA
K = 10
sX_train = 1/(np.std(X_train, axis = 0) + 10**(-10)) * (X_train - np.mean(X_train, axis = 0))
E_train = 1/(sX_train.shape[0] - 1) * sX_train.T @ sX_train
spectrum_train, U_train = np.linalg.eig(E_train)
I = np.argsort(-spectrum_train)[0:K]
Z_train = sX_train @ U[:, I]
sX_test = 1/(np.std(X_test, axis = 0) + 10**(-10)) * (X_test - np.mean(X_test, axis = 0))
Z_test = sX_test @ U[:, I]

lr = LogisticRegression(solver = "lbfgs", max_iter = 1000, multi_class = "multinomial")
lr.fit(Z_train, y_train)
print("Train accuracy for K =", K, ":", 1/Z_train.shape[0] * np.sum(lr.predict(Z_train) == y_train))
print("Test accuracy for K =", K, ":", 1/Z_test.shape[0] * np.sum(lr.predict(Z_test) == y_test))

