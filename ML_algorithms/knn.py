import numpy as np
import matplotlib.pyplot as plt

class KNN():
  def __init__(self, k):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    pred = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
      distances = np.zeros(X_train.shape[0])
      for j in range(0, self.X_train.shape[0]):
        distances[j] = np.sqrt(np.sum(np.power(X[i, :] - X_train[j, :], 2)))
      k_nearest = y_train[np.argsort(distances)[0:self.k]]
      k_nearest_set, I = np.unique(k_nearest, return_inverse=True)
      mode = k_nearest_set[np.argmax(np.bincount(I))]
      pred[i] = mode
    return pred

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

# Train/test split
n_samples = X.shape[0]
n_TRAIN = int(.75 * n_samples)
I = np.arange(0, n_samples)
TRAIN = np.random.choice(I, n_TRAIN, replace = False)
TEST = np.setdiff1d(I, TRAIN)
X_train = X[TRAIN, :]
y_train = y[TRAIN]
X_test = X[TEST, :]
y_test = y[TEST]

# Fit
K = [1, 2, 5]
accuracies = np.array([])
for k in K:
  classifier = KNN(k = k)
  classifier.fit(X_train, y_train)
  accuracy = 1/X_test.shape[0] * np.sum(classifier.predict(X_test) == y_test)
  accuracies = np.append(accuracies, accuracy)
  print("Test accuracy for k =", k, ":", accuracy)

# Plot accuracy vs K
plt.style.use("default")
fig = plt.figure(figsize=(8, 6))
plt.plot(K, accuracies, marker = 'o')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
