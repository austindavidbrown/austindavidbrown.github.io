import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn import svm


###
# Opdigits test dataset
###
test = np.loadtxt("data/optdigits_test.txt", delimiter = ",")
X = test[:, 0:64]
y = test[:, 64]

def cvK(X, y, tuning_params, partitions, k):
  n_tuning_params = tuning_params.shape[0]

  partition = partitions[k]
  TRAIN = np.delete(np.arange(0, X.shape[0]), partition)
  TEST = partition
  X_train = X[TRAIN, :]
  y_train = y[TRAIN]

  X_test = X[TEST, :]
  y_test = y[TEST]

  accuracies = np.zeros(n_tuning_params)
  for i in range(0, n_tuning_params):
    svc = svm.SVC(C = tuning_params[i], kernel = "linear")
    accuracies[i] = svc.fit(X_train, y_train).score(X_test, y_test)
  return accuracies

K = 5
tuning_params = np.logspace(-6, -1, 10)
partitions = np.array_split(np.random.permutation([i for i in range(0, X.shape[0])]), K)

pool = Pool()
args = [(X, y, tuning_params, partitions, k) for k in range(0, K)]
Accuracies = np.array(pool.starmap(cvK, args))
pool.close()

CV_accuracy = np.mean(Accuracies, axis = 0)
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
print(best_tuning_param)
