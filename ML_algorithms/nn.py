import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

class NN():
  def __init__(self, n_samples, n_features, n_outputs, n_hidden = 1):
    self.n_samples = n_samples
    self.n_features = n_features
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs

    self.W_h = np.random.randn(n_features, n_hidden)
    self.b_h = .01 + np.zeros((1, n_hidden))
    self.W_o = np.random.randn(n_hidden, n_outputs)
    self.b_o = .01 + np.zeros((1, n_outputs))

  def sigmoid(self, x):
    return 1/(1 + np.exp(-x))

  def loss(self, y, p_pred):
    return -1/y.shape[0] * (np.sum(y * np.log(p_pred) + (1 - y) * (np.log(1 - p_pred))))

  def predict(self, X):
    return np.squeeze(np.round(self.forward_prop(X)["O"]))

  def forward_prop(self, X):
    # Hidden layer
    A_h = X @ self.W_h + self.b_h
    H = self.sigmoid(A_h)

    # Relu
    #H = np.copy(A_h)
    #H[H > 0] = 0

    # Output layer
    A_o = H @ self.W_o + self.b_o
    O = self.sigmoid(A_o)
    return {
      "A_h": A_h, 
      "H": H, 
      "A_o": A_o, 
      "O": O
    }

  # This is not a true implmentation of backprop
  def backward_prop(self, X, y_, forward):
    one_n = np.ones(self.n_samples)
    y = (y_[np.newaxis]).T # convert to column vector

    dA_o = (y - forward["O"])
    dL_dW_o = 1/self.n_samples * forward["H"].T @ dA_o
    dL_db_o = 1/self.n_samples * one_n.T @ dA_o
    
    dA_h = (dA_o @ self.W_o.T) * (self.sigmoid(forward["A_h"]) * (1 - self.sigmoid(forward["A_h"])))
    # dA_h = (dA_o @ self.W_o.T) * (1 * (forward["A_h"] > 0)) # Relu
    dL_dW_h = 1/self.n_samples * X.T @ dA_h
    dL_db_h = 1/self.n_samples * one_n.T @ dA_h

    return {
      "dL_dW_h": dL_dW_h, 
      "dL_db_h": dL_db_h, 
      "dL_dW_o": dL_dW_o, 
      "dL_db_o": dL_db_o
    }

  def train(self, X, y, learning_rate = .5, max_iter = 1001):
    for i in range(0, max_iter):
      forward_prop_dict = self.forward_prop(X)
      G = self.backward_prop(X, y, forward_prop_dict)

      # Gradient step
      self.W_h = self.W_h + learning_rate * G["dL_dW_h"]
      self.b_h = self.b_h + learning_rate * G["dL_db_h"]

      self.W_o = self.W_o + learning_rate * G["dL_dW_o"]
      self.b_o = self.b_o + learning_rate * G["dL_db_o"]

      if i % 100 == 0:
        print(f"Iteration: {i}, Training Loss: {self.loss(y, np.squeeze(forward_prop_dict['O']))}")

n_samples = 1000
n_features = 2
n_ouputs = 1
X, y = make_circles(n_samples = n_samples, factor = .01, noise = .2)

n_TRAIN = int(.75 * n_samples)
X_train = X[0:n_TRAIN, :]
y_train = y[0:n_TRAIN]
X_test = X[n_TRAIN:n_samples, :]
y_test = y[n_TRAIN:n_samples]

plt.figure(figsize=(8, 8))
plt.scatter(X[:,0], X[:,1], c = y, s = 2)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

nn = NN(n_samples = n_TRAIN, n_features = n_features, n_outputs = n_ouputs, n_hidden = 10)
nn.train(X_train, y_train)

print("Train accuracy:", 1/X_train.shape[0] * np.sum(nn.predict(X_train) == y_train))
print("Test accuracy:", 1/X_test.shape[0] * np.sum(nn.predict(X_test) == y_test))