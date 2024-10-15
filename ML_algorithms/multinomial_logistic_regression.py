import numpy as np
import matplotlib.pyplot as plt

class MultRegression():
  @staticmethod
  def softmax(M):
    exps = np.exp(M)
    S_exps = (exps @ np.ones(M.shape[1]))[:, np.newaxis]
    return 1/S_exps * exps

  @staticmethod
  def entropy(Y, P):
    n_samples = P.shape[0]
    n_classes = P.shape[1]
    return -1/n_samples * np.ones(n_samples).T @ (Y * np.log(P) @ np.ones(n_classes))
  
  def train(self, X, y, max_iter = 501, learning_rate = .1):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = np.unique(y).shape[0]

    # Convert to a multinomial vector
    Y = np.zeros((n_samples, n_classes))
    Y[np.arange(n_samples), np.array(y.T, dtype = int)] = 1

    self.W = np.zeros((n_features, n_classes))
    self.b = np.zeros((1, n_classes))
    for i in range(0, max_iter):
      # Forward pass
      A_o = X @ self.W + self.b * np.ones((n_samples, n_classes))
      O = MultRegression.softmax(A_o)
      if i % 100 == 0:
        print(f"Iteration: {i}, Training Loss: {MultRegression.entropy(Y, O)}")

      # Backward pass
      dW = 1/n_samples * X.T @ (O - Y)
      db = 1/n_samples * np.ones((n_samples, 1)).T @ (O - Y)

      # Gradient step
      self.W = self.W - learning_rate * dW
      self.b = self.b - learning_rate * db

  def predict(self, X):
    n_samples = X.shape[0]
    n_classes = self.W.shape[1]

    A_o = X @ self.W + self.b * np.ones((n_samples, n_classes))
    O = MultRegression.softmax(A_o)
    return np.argmax(O, axis = 1)

###
# Opdigits test dataset
###
test = np.loadtxt("data/optdigits_test.txt", delimiter = ",")
X = test[:, 0:64]
y = test[:, 64]

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

mlr = MultRegression()
mlr.train(X_train, y_train)

print("Train accuracy:", 1/X_train.shape[0] * np.sum((mlr.predict(X_train) == y_train).astype(int)))
print("Test accuracy:", 1/X_test.shape[0] * np.sum((mlr.predict(X_test) == y_test).astype(int)))


###
# Blobs
###
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=4, n_samples = 5000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

W, b = MultRegression.train(X_train, y_train)
print("Train accuracy:", 1/X_train.shape[0] * np.sum(MultRegression.predict(X_train, W, b) == y_train))
print("Test accuracy:", 1/X_test.shape[0] * np.sum(MultRegression.predict(X_test, W, b) == y_test))
