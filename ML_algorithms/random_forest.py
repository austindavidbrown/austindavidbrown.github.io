import numpy as np
import matplotlib.pyplot as plt


class RFTree():
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
  def greedy_best_split(X, y, m_features):
    best_feature_index = 0
    best_split_value = 0
    best_IG = 0
    best_split = {
      "I_left": np.array([]),
      "I_right": np.array([]),
    }

    parent_entropy = RFTree.entropy(y)
    N = y.shape[0]

    # Subsample m features and determine the optimal split using this subset.
    n_features = X.shape[1]
    feature_index_subset = np.random.choice(n_features, m_features, replace = False)
    for feature_index in feature_index_subset:
      split_values = np.unique(X[:, feature_index])
      for split_value in split_values:
        split = RFTree.split_data(X, y, feature_index, split_value)

        # Compute the information gain
        N_left = split["I_left"].shape[0]
        N_right = split["I_right"].shape[0]
        IG = parent_entropy - 1/N * (N_left * RFTree.entropy(y[split["I_left"]]) + N_right * RFTree.entropy(y[split["I_right"]]))

        # Update if the information gain is the largest so far
        if IG >= best_IG:
          best_feature_index = feature_index
          best_split_value = split_value
          best_split = split
          best_IG = IG
    return best_IG, best_feature_index, best_split_value, best_split

  @staticmethod
  def fit_tree(X, y, m_features, depth = 1,
               max_depth = 100, tolerance = 10**(-3)):
    node = {}

    # If we can split, find the best split by greedy algorithm
    if y.shape[0] >= 2:
      IG, feature_index, split_value, split = RFTree.greedy_best_split(X, y, m_features)
      # If there is a greedy split and the stopping criterion is not met, branch 2 times
      if split["I_left"].shape[0] > 0 and split["I_right"].shape[0] > 0 and IG >= tolerance and depth < max_depth:
        node["information_gain"] = IG
        node["feature_index"] = feature_index
        node["split_value"] = split_value

        node["left"] = RFTree.fit_tree(X[split["I_left"]], y[split["I_left"]], m_features, depth = depth + 1, 
                                      max_depth = max_depth, tolerance = tolerance)
        node["right"] = RFTree.fit_tree(X[split["I_right"]], y[split["I_right"]], m_features, depth = depth + 1, 
                                       max_depth = max_depth, tolerance = tolerance) 
      else:
        # Set weight with the mode
        S_y, counts = np.unique(y, return_counts = True)
        node["w"] = S_y[np.argmax(counts)] # mode

        node["left"] = None
        node["right"] = None
    else:
      # Set weight with the mode
      S_y, counts = np.unique(y, return_counts = True)
      node["w"] = S_y[np.argmax(counts)] # mode

      node["left"] = None
      node["right"] = None
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
        return RFTree.predict_one(node["left"], x)
      else:
        return RFTree.predict_one(node["right"], x)

  @staticmethod
  def predict(node, X):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    for i in range(0, n_samples):
      predictions[i] = RFTree.predict_one(node, X[i])
    return predictions

  @staticmethod
  def print_tree(node, depth = 0):
    if node["left"] == None:
      print(f'{depth * "  "}weight: {node["w"]}')
    else:
      print(f'{depth * "  "}X{node["feature_index"]} <= {node["split_value"]}')
      RFTree.print_tree(node["left"], depth + 1)
      RFTree.print_tree(node["right"], depth + 1)


class RandomForest():
  def __init__(self, n_boot = 500, max_depth = 10, tolerance = 10**(-3)):
    self.trees = []
    self.n_boot = n_boot
    self.max_depth = max_depth
    self.tolerance = tolerance

  def train(self, X, y, m_features = 0,):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Default to \sqrt{n_{features}} subsampled features for each tree
    if m_features == 0:
      m_features = int(np.floor(np.sqrt(n_features))) # features to subsample

    # Construct sequence of trees
    for b in range(0, self.n_boot):
      I = np.random.choice(n_samples, n_samples, replace = True) # bootstrap sample
      X_B = X[I, :]
      y_B = y[I]

      tree = RFTree.fit_tree(X_B, y_B, m_features, max_depth = self.max_depth, tolerance = self.tolerance)
      self.trees.append(tree)
      if b % 10 == 0:
        print("Trained tree:", b)

  def predict(self, X):
    n_samples = X.shape[0]
    n_trees = len(self.trees)

    # Construct matrix of all tree predictions: rows are samples, columns are trees
    pred_matrix = np.zeros((n_samples, n_trees), dtype = int)
    for i in range(0, n_trees):
      pred_matrix[:, i] = RFTree.predict(self.trees[i], X)

    # Predict with the mode
    y_pred = np.zeros(n_samples, dtype = int)
    for i in range(0, n_samples):
      S_y_pred, counts = np.unique(pred_matrix[i, :], return_counts = True)
      y_pred[i] = S_y_pred[np.argmax(counts)]

    return y_pred

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

rf = RandomForest(n_boot = 100)
rf.train(X_train, y_train)

print( "Train accuracy:", 1/X_train.shape[0] * np.sum((rf.predict(X_train) == y_train).astype(int)) )
print( "Test accuracy", 1/X_test.shape[0] * np.sum((rf.predict(X_test) == y_test).astype(int)) )


