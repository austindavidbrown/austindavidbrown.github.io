import numpy as np
import matplotlib.pyplot as plt

###
# Rejection Sampling
###
def rejectionSampler(target, N):
  Y = np.zeros(N)
  i = 0
  while i < N:
    X = 10 * np.random.uniform(low = -5, high = 5)
    u = np.random.uniform()
    if u <=  target(X):
      Y[i] = X
      i += 1
  return Y

def target(x):
  return 1/np.sqrt(2 * np.pi) * np.exp(-1/2 * x**2)

X = rejectionSampler(target, 10000)
plt.figure(figsize = (8, 8))
plt.tight_layout()
plt.hist(X, normed = True, bins = 100)
plt.show()

###
# M-H Sampling
###
def randomWalkMH(target, dim, N = 10**(4), h = 1):
  X = np.zeros((N, dim))
  X[0] = np.repeat(0, dim)
  for i in range(0, N - 1):
    Z = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
    X_next = X[i] + h * Z

    rho = min(1, target(X_next) / max(10**(-20), target(X[i])))
    u = np.random.uniform()
    if u < rho:
      X[i + 1] = X_next
    else:
      X[i + 1] = X[i]
  return X


def target(x):
  mu1 = np.array([-5, 0])
  mu2 = np.array([5, 0])
  f1 = 1/np.sqrt(2 * np.pi)**2 * np.exp(-1/2 * np.linalg.norm(x - mu1)**2)
  f2 = 1/np.sqrt(2 * np.pi)**2 * np.exp(-1/2 * np.linalg.norm(x - mu2)**2)
  return 1/2 * f1 + 1/2 * f2

dim = 2
X = randomWalkMH(target, dim, N = 10**(5), h = 5)
plt.figure(figsize = (10, 5))
plt.tight_layout()
plt.scatter(X[:, 0], X[:, 1], s=1)
plt.show()


###
# Gibbs Sampling
###

def bivariateNormalgibbsSampler(mu, E, N = 1000):
  X = np.zeros((N, 2))
  for i in range(0, N - 1):
    X[i + 1][0] = np.random.normal(E[0, 1] * 1/E[1, 1] * (X[i][1] - mu[1]), E[1, 1] - E[0, 1] * E[1, 0])
    X[i + 1][1] = np.random.normal(E[1, 0] * 1/E[0, 0] * (X[i + 1][0] - mu[1]), E[0, 0] - E[0, 1] * E[1, 0])
  return X

mu = np.array([0, 0])
E = np.matrix([[1, -1/2], [-1/2, 1]])
X = bivariateNormalgibbsSampler(mu, E, N = 10**(4))
plt.figure(figsize = (8, 6))
plt.tight_layout()
plt.scatter(X[:, 0], X[:, 1], s=1)
plt.savefig('tmp.png', bbox_inches='tight')
plt.show()
