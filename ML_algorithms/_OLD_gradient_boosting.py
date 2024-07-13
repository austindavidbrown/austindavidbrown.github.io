import numpy as np
import matplotlib.pyplot as plt

###
# Generate regression data
###
n_samples = 100
n_features = 10
w_0 = 5 * np.ones(n_samples)
w = 3 * np.ones(n_features)

X = np.zeros((n_samples, n_features))
for i in range(0, n_samples):
  X[i, :] = np.random.multivariate_normal(np.zeros(n_features), 
                                          np.identity(n_features))
y = w_0 + X @ w


# Fit intercept
w_0 = 1/(np.ones(n_samples).T @ np.ones(n_samples)) * np.ones(n_samples).T @ y
# Fit weights
w = np.zeros(n_features)
U = np.arange(0, n_features)
for j in range(0, n_features):
  ###
  # Greedy search for the best feature to step in
  ###
  best_w_k = -1
  best_k = -1
  best_loss = np.inf
  for k in U:
    X_k = X[:, k]
    w_k = 1/(X_k.T @ X_k) * X_k.T @ (y - w_0 * np.ones(n_samples) - X @ w)
    loss = (y - w_0 * np.ones(n_samples) - X @ w - X_k * w_k) @ (y - w_0 * np.ones(n_samples) - X @ w - X_k * w_k)
    if loss <= best_loss:
      best_w_k = w_k
      best_k = k
      best_loss = loss
  U = np.delete(U, np.where(U == best_k)[0][0])    

  # line search
  loss_old = (y - w_0 * np.ones(n_samples) - X @ w) @ (y - w_0 * np.ones(n_samples) - X @ w)
  s = 1
  while True:
    loss_new = (y - w_0 * np.ones(n_samples) - X @ w - s * X[:, best_k] * best_w_k) @ (y - w_0 * np.ones(n_samples) - X @ w - s * X[:, best_k] * best_w_k)
    if loss_old - loss_new  >= 0 or s == 0:
      # update weight with best step size
      w[best_k] = s * best_w_k
      break
    else:
      s = 1/2 * s

