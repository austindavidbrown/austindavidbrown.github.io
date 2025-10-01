import numpy as np
import matplotlib.pyplot as plt


class DTree():
  @staticmethod
  def entropy(v):
    S, counts = np.unique(v, return_counts = True)
    N = v.shape[0]
    p = counts / N
    return -np.sum(np.log2(p) * p)

  @staticmethod
  def split_data(X, y, feature_index, feature_value):
    return {
      "I_left": np.where(X[:, feature_index] <= feature_value)[0],
      "I_right": np.where(X[:, feature_index] > feature_value)[0],
    }

  # Greedy algorithm for finding the best split using information gain
  # We look for the split with the best increase in information gain
  @staticmethod
  def greedy_best_split(X, y):
    best_feature_index = 0
    best_split_value = 0
    best_IG = 0
    best_split = {
      "I_left": np.array([]),
      "I_right": np.array([]),
    }

    n_features = X.shape[1]
    parent_entropy = DTree.entropy(y)
    N = y.shape[0]
    for feature_index in range(0, n_features):
      split_values = np.unique(X[:, feature_index])
      for split_value in split_values:
        split = DTree.split_data(X, y, feature_index, split_value)

        # Compute the information gain
        N_left = split["I_left"].shape[0]
        N_right = split["I_right"].shape[0]
        IG = parent_entropy - 1/N * (N_left * DTree.entropy(y[split["I_left"]]) + N_right * DTree.entropy(y[split["I_right"]]))

        # Update if the information gain is the largest so far
        if IG >= best_IG:
          best_feature_index = feature_index
          best_split_value = split_value
          best_split = split
          best_IG = IG
    return best_IG, best_feature_index, best_split_value, best_split

  @staticmethod
  def fit_tree(X, y, depth = 1, 
               max_depth = 100, tolerance = 10**(-3)):
    node = {}

    # Set weight with the mode
    S_y, counts = np.unique(y, return_counts = True)
    node["w"] = S_y[np.argmax(counts)] # mode

    node["left"] = None
    node["right"] = None

    # If we can split, find the best split by greedy algorithm
    if y.shape[0] >= 2:
      IG, feature_index, split_value, split = DTree.greedy_best_split(X, y)
      # If there is a greedy split and the stopping criterion is not met, branch 2 times
      if split["I_left"].shape[0] > 0 and split["I_right"].shape[0] > 0 and IG >= tolerance and depth < max_depth:
        node["information_gain"] = IG
        node["feature_index"] = feature_index
        node["split_value"] = split_value

        node["left"] = DTree.fit_tree(X[split["I_left"]], y[split["I_left"]], depth = depth + 1, max_depth = max_depth, tolerance = tolerance)
        node["right"] = DTree.fit_tree(X[split["I_right"]], y[split["I_right"]], depth = depth + 1, max_depth = max_depth, tolerance = tolerance) 
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
        return DTree.predict_one(node["left"], x)
      else:
        return DTree.predict_one(node["right"], x)

  @staticmethod
  def predict(node, X):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    for i in range(0, n_samples):
      predictions[i] = DTree.predict_one(node, X[i])
    return predictions

  @staticmethod
  def print_tree(node, depth = 0):
    if node["left"] == None:
      print(f'{depth * "  "}weight: {node["w"]}')
    else:
      print(f'{depth * "  "}X{node["feature_index"]} <= {node["split_value"]}')
      DTree.print_tree(node["left"], depth + 1)
      DTree.print_tree(node["right"], depth + 1)


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

tree = DTree.fit_tree(X_train, y_train, max_depth = 100, tolerance = 10**(-3))
DTree.print_tree(tree)

print("Train accuracy:", 1/X_train.shape[0] * np.sum(DTree.predict(tree, X_train) == y_train))
print("Test accuracy", 1/X_test.shape[0] * np.sum(DTree.predict(tree, X_test) == y_test))


###
# Iris dataset
###
import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", 
                 header = None, 
                 names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "labels"])
df["labels"] = pd.Categorical(df["labels"]).codes
data = df.values
X = data[:, 0:4]
y = data[:, 4]

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

tree = DTree.fit_tree(X_train, y_train, max_depth = 100, tolerance = 10**(-2))
DTree.print_tree(tree)
print("Train accuracy:", 1/X_train.shape[0] * np.sum(DTree.predict(tree, X_train) == y_train))
print("Test accuracy", 1/X_test.shape[0] * np.sum(DTree.predict(tree, X_test) == y_test))


