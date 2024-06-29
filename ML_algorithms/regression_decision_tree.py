import numpy as np
import matplotlib.pyplot as plt


class RegTree():
  @staticmethod
  def mse(v):
    return np.mean(np.square(v - np.mean(v)))

  @staticmethod
  def split_data(X, y, feature_index, feature_value):
    return {
      "I_left": np.where(X[:, feature_index] <= feature_value)[0],
      "I_right": np.where(X[:, feature_index] > feature_value)[0],
    }

  # Greedy algorithm for finding the best split
  @staticmethod
  def greedy_best_split(X, y):
    best_feature_index = 0
    best_split_value = 0
    best_dloss = 0
    best_split = {
      "I_left": np.array([]),
      "I_right": np.array([]),
    }

    n_features = X.shape[1]
    parent_mse = RegTree.mse(y)
    N = y.shape[0]
    for feature_index in range(0, n_features):
      split_values = np.unique(X[:, feature_index])
      for split_value in split_values:
        split = RegTree.split_data(X, y, feature_index, split_value)

        # If there is a split
        if split["I_left"].shape[0] > 0 and split["I_right"].shape[0] > 0:
          # Compute the change in loss
          N_left = split["I_left"].shape[0]
          N_right = split["I_right"].shape[0]
          dloss = parent_mse - 1/N * (N_left * RegTree.mse(y[split["I_left"]]) + N_right * RegTree.mse(y[split["I_right"]]))

          # Update if the change in loss is the largest so far
          if dloss >= best_dloss:
            best_feature_index = feature_index
            best_split_value = split_value
            best_split = split
            best_dloss = dloss

    return best_dloss, best_feature_index, best_split_value, best_split

  @staticmethod
  def fit_tree(X, y, depth = 1, 
               max_depth = 100, tolerance = 10**(-3)):
    node = {}

    # Predict with the mean
    node["w"] = np.mean(y)

    node["left"] = None
    node["right"] = None

    # If we can split, find the best split by greedy algorithm
    if y.shape[0] >= 2:
      dloss, feature_index, split_value, split = RegTree.greedy_best_split(X, y)
      # If there is a greedy split and the stopping criterion is not met, branch 2 times
      if split["I_left"].shape[0] > 0 and split["I_right"].shape[0] > 0 and dloss >= tolerance and depth < max_depth:
        node["dloss"] = dloss
        node["feature_index"] = feature_index
        node["split_value"] = split_value

        node["left"] = RegTree.fit_tree(X[split["I_left"]], y[split["I_left"]], depth = depth + 1, max_depth = max_depth, tolerance = tolerance)
        node["right"] = RegTree.fit_tree(X[split["I_right"]], y[split["I_right"]], depth = depth + 1, max_depth = max_depth, tolerance = tolerance) 
    return node

  ###
  # Predict
  ###
  @staticmethod
  def predict_one(node, x):
    if node["left"] == None:
      return node["w"]
    else:
      if x[node["feature_index"]] <= node["split_value"]:
        return RegTree.predict_one(node["left"], x)
      else:
        return RegTree.predict_one(node["right"], x)

  @staticmethod
  def predict(node, X):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    for i in range(0, n_samples):
      predictions[i] = RegTree.predict_one(node, X[i])
    return predictions

  @staticmethod
  def print_tree(node, depth = 0):
    if node["left"] == None:
      print(f'{depth * "  "}weight: {node["w"]}')
    else:
      print(f'{depth * "  "}X{node["feature_index"]} <= {node["split_value"]}')
      RegTree.print_tree(node["left"], depth + 1)
      RegTree.print_tree(node["right"], depth + 1)



###
# Generate regression data
###
n_samples = 100
n_features = 10
intercept = 5 * np.ones(n_samples)
B = 3 * np.ones(n_features)

X = np.zeros((n_samples, n_features))
for i in range(0, n_samples):
  X[i, :] = np.random.multivariate_normal(np.zeros(n_features), 10 * np.identity(n_features))
e = np.random.multivariate_normal(np.zeros(n_samples), np.identity(n_samples))
y = intercept + X @ B + e

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


tree = RegTree.fit_tree(X_train, y_train, max_depth = 100, tolerance = 10**(-3))
# RegTree.print_tree(tree)
print("Train MSE:", 1/X_train.shape[0] * np.sum(np.square(y_train - RegTree.predict(tree, X_train))))
print("Train MSE:", 1/X_test.shape[0] * np.sum(np.square(y_test - RegTree.predict(tree, X_test))))

