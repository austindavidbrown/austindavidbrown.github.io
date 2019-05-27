import math
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt

torch.no_grad() 
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

def LMC(N, h, X_0):
  X = torch.zeros(N, 2)
  X[0] = X_0

  for i in range(0, N - 1):
    Z = torch.normal(torch.zeros(2), 1)
    Y = X[i] - h * X[i] + torch.sqrt(torch.tensor(2 * h)) * Z

    u = torch.zeros(1).uniform_(0, 1).item()

    R = (1/2.0 * (torch.norm(X[i], p = 2)**2 - torch.norm(Y, p = 2)**2)) + 1/(4.0 * h) * (torch.norm(Y - 1 * X[i] + h * X[i], p = 2)**2 - torch.norm(X[i] - Y + h * Y, p = 2)**2)
    a = min(1, torch.exp(R).item())
    if u <= a:
      X[i + 1] = Y
    else:
      X[i + 1] = X[i]
  return X

# setup
N = 10**3
h = .1
X_0 = torch.zeros(2).fill_(3)

X = LMC(N, h, X_0)
print(torch.mean(X, 0))

"""
plt.style.use("seaborn")
plt.figure()
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, s=1)
plt.show()
"""

def bmSE(X, M):
  N = X.size(0)
  M = M # how many batches
  B = int(N/M) # batch size

  S2s = []
  mu = torch.mean(X.narrow(0, 0, B * M), 0)
  for m in range(0, M):
    B_m = torch.mean(X.narrow(0, m * B, B), 0)
    S2_m = torch.pow(B_m - mu, 2)
    S2s.append(S2_m)
  se = torch.sqrt(B * torch.mean(torch.stack(S2s), 0))
  return se

torch.set_default_dtype(torch.float64)
torch.no_grad() 
torch.manual_seed(0)

N = 1000
h = .1
X_0 = torch.zeros(2).fill_(3)
X = LMC(N, h, X_0)
se = bmSE(X, 100)

for i in range(0, 100):
  X = LMC(N, h, X[N - 1, :])
  se_old = se
  se = bmSE(X, int(X.size(0)/10))
  print(se)
  if torch.all(torch.abs(se - se_old) < .05):
    print("Converged at i =", i)
    break




   