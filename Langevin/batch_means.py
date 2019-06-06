import torch
from torch.distributions import Normal
from math import sqrt

torch.no_grad() 
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

def OULMC(N, h, X_0):
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
  se = torch.sqrt(M * torch.mean(torch.stack(S2s), 0))
  return se

N = 10**4
h = .1
X_0 = torch.zeros(2).fill_(3)
X = OULMC(N, h, X_0)


# CI
t = 1.660234
mu = torch.mean(X, 0)
se = bmSE(X, int(sqrt(N)))
M = int(sqrt(N))
print(mu + se * t * 1/sqrt(M))
print(mu - se * t * 1/sqrt(M))


###
# Stopping criteria
###
torch.manual_seed(0)

X = OULMC(N, h, X_0)
se = bmSE(X, int(sqrt(N)))

for i in range(0, 100):
  X_new = OULMC(N, h, X[N - 1, :])
  X = torch.cat([X, X_new], 0)

  se_old = se
  se = bmSE(X, int(sqrt(N)))
  print(se)
  if torch.all(torch.abs(se - se_old) < .1):
    print("Converged at i =", i + 2)
    break

# CI
t = 1.646761
mu = torch.mean(X, 0)
se = bmSE(X, int(sqrt(N)))
M = 8 * int(sqrt(N))
print(mu + se * t * 1/sqrt(M))
print(mu - se * t * 1/sqrt(M))


"""
import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.figure()
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, s=1)
plt.show()
"""



   