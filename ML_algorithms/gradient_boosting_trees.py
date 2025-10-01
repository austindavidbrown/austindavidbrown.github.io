import numpy as np
import matplotlib.pyplot as plt

class BinaryEntropy():
  @staticmethod
  def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

  # gradient
  @staticmethod
  def g(y_1, y_2):
    return BinaryEntropy.sigmoid(y_2) - y_1

  # Hessian diagonal
  @staticmethod
  def H_diag(y_1, y_2):
    return BinaryEntropy.sigmoid(y_2) * (1 - BinaryEntropy.sigmoid(y_2))

  @staticmethod
  def compute(y, p_pred):
    return -1/y.shape[0] * (np.sum(y * np.log(p_pred) + (1 - y) * (np.log(1 - p_pred))))

  @staticmethod
  def approx(y, y_pred):
    eps = 10**(-16)
    return -1/2 * 1/max(np.sum(BinaryEntropy.H_diag(y, y_pred)), eps) * np.sum(BinaryEntropy.g(y, y_pred))**2

  @staticmethod
  def compute_weight(y, y_pred):
    eps = 10**(-16)
    return -1/max(np.sum(BinaryEntropy.H_diag(y, y_pred)), eps) * np.sum(BinaryEntropy.g(y, y_pred))

  @staticmethod
  def predTransform(w):
    return BinaryEntropy.sigmoid(w)


class GBTree():
  @staticmethod
  def split_data(X, y, feature_index, feature_value):
    return {
      "I_left": np.where(X[:, feature_index] <= feature_value)[0],
      "I_right": np.where(X[:, feature_index] > feature_value)[0],
    }

  @staticmethod
  def greedy_best_split(X, y, y_pred, loss):
    best_feature_index = 0
    best_split_value = 0
    best_dloss = 0
    best_split = {
      "I_left": np.array([]),
      "I_right": np.array([]),
    }

    n_features = X.shape[1]
    parent_loss = loss.approx(y, y_pred)
    for feature_index in range(0, n_features):
      split_values = np.unique(X[:, feature_index])
      for split_value in split_values:
        split = GBTree.split_data(X, y, feature_index, split_value)

        # If there is a split
        if split["I_left"].shape[0] > 0 and split["I_right"].shape[0] > 0:
          # Compute loss change and update if the change in loss is the best so far
          dloss = parent_loss - loss.approx(y[split["I_left"]], y_pred[split["I_left"]]) - loss.approx(y[split["I_right"]], y_pred[split["I_right"]])
          if dloss >= best_dloss:
            best_feature_index = feature_index
            best_split_value = split_value
            best_split = split
            best_dloss = dloss
    return best_dloss, best_feature_index, best_split_value, best_split

  @staticmethod
  def fit_tree(X, y, y_pred, loss, depth = 1, 
               max_depth = 100, tolerance = 10**(-3)):
    node = {}

    # If we can split, find the best split by greedy algorithm
    if y.shape[0] >= 2:
      dloss, feature_index, split_value, split = GBTree.greedy_best_split(X, y, y_pred, loss)
      node["dloss"] = dloss
      node["feature_index"] = feature_index
      node["split_value"] = split_value

      # If there is a split and the stopping criterion is not met, branch 2 leaves
      if split["I_left"].shape[0] > 0 and split["I_right"].shape[0] > 0 and dloss >= tolerance and depth < max_depth:
        node["left"] = GBTree.fit_tree(X[split["I_left"]], y[split["I_left"]], y_pred[split["I_left"]], loss,
                                     depth = depth + 1, max_depth = max_depth, tolerance = tolerance)
        node["right"] = GBTree.fit_tree(X[split["I_right"]], y[split["I_right"]], y_pred[split["I_right"]], loss,
                                      depth = depth + 1, max_depth = max_depth, tolerance = tolerance) 
      # Terminal node
      else:
        node["w"] = loss.compute_weight(y, y_pred)
        node["left"] = None
        node["right"] = None
    # Terminal node    
    else:
      node["w"] = loss.compute_weight(y, y_pred)
      node["left"] = None
      node["right"] = None
    return node

  @staticmethod
  def predict_one(node, x):
    if node["left"] == None or node["right"] == None:
      return node["w"]
    else:
      if x[node["feature_index"]] <= node["split_value"]:
        return GBTree.predict_one(node["left"], x)
      else:
        return GBTree.predict_one(node["right"], x)

  @staticmethod
  def predict(node, loss, X):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    for i in range(0, n_samples):
      predictions[i] = GBTree.predict_one(node, X[i])
    return predictions

  @staticmethod
  def print_tree(node, depth = 0):
    if node["left"] == None or node["right"] == None:
      print(f'{depth * "  "}weight: {node["w"]}')
    else:
      print(f'{depth * "  "}X{node["feature_index"]} <= {node["split_value"]}')
      GBTree.print_tree(node["left"], depth + 1)
      GBTree.print_tree(node["right"], depth + 1)


class GradientBoost():
  def __init__(self, loss, learning_rate = 1, max_depth = 10, tolerance = 10**(-3), max_iter = 10):
    self.trees = []
    self.loss = loss
    self.learning_rate = learning_rate
    self.max_depth = max_depth
    self.tolerance = tolerance
    self.max_iter = max_iter

  def train(self, X, y):
    y_pred = np.zeros(y.shape[0])

    for i in range(0, self.max_iter):
      tree = GBTree.fit_tree(X, y, y_pred, self.loss,
                           max_depth = self.max_depth, tolerance = self.tolerance)
      self.trees.append(tree)
      y_pred = self.learning_rate * GBTree.predict(tree, self.loss, X)

      print("Training Loss:", self.loss.compute(y, self.predict(X)))

  def predict(self, X):
    n_samples = X.shape[0]
    
    y_pred = np.zeros(n_samples)
    for i in range(0, len(self.trees)):
      y_pred += self.learning_rate * GBTree.predict(self.trees[i], self.loss, X) 
    return self.loss.predTransform(y_pred)



###
# Opdigits test dataset
###
test = np.loadtxt("data/optdigits_test.txt", delimiter = ",")
X = test[:, 0:64]
y = test[:, 64]

# Convert to binary classification
y[y != 1] = 0

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

gb = GradientBoost(loss = BinaryEntropy(), learning_rate = 1, max_depth = 10, max_iter = 10)
gb.train(X_train, y_train)

print("Train accuracy:", 1/X_train.shape[0] * np.sum((np.round(gb.predict(X_train)) == y_train).astype(int)))
print("Test accuracy", 1/X_test.shape[0] * np.sum((np.round(gb.predict(X_test)) == y_test).astype(int)))


"""
y_pred = np.zeros(y.shape[0])
tree = GBTree.fit_tree(X, y, y_pred, loss = BinaryEntropy(), max_depth = 100, tolerance = 10**(-3))
GBTree.print_tree(tree)
GBTree.predict(tree, loss = BinaryEntropy(), X)
"""


