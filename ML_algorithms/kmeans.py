import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Make clusters
X, y = make_blobs(n_samples=1000, centers = 4)
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c = y, s = 3)
# Generate html image
plt.savefig('tmp1.png', bbox_inches='tight')
import base64
data_uri = base64.b64encode(open('tmp1.png', 'rb').read()).decode('utf-8').replace('\n', '')
img_tag = '<img height="300" src="data:image/png;base64,%s">' % data_uri
with open("tmp1.txt", "w") as text_file:
    text_file.write(img_tag)

# Lloyds algorithm for kmeans
def kmeans(X, k, max_iter = 100, tolerance = 10**(-3)):
  n_samples = X.shape[0]
  n_features = X.shape[1]
  classifications = np.zeros(n_samples, dtype = np.int64)

  # Choose initial cluster centroids randomly
  I = np.random.choice(n_samples, k)
  centroids = X[I, :]

  loss = 0
  for m in range(0, max_iter):
    # Compute the classifications
    for i in range(0, n_samples):
      distances = np.zeros(k)
      for j in range(0, k):
        distances[j] = np.sqrt(np.sum(np.power(X[i, :] - centroids[j], 2))) 
      classifications[i] = np.argmin(distances)

    # Compute the new centroids and new loss
    new_centroids = np.zeros((k, n_features))
    new_loss = 0
    for j in range(0, k):
      # compute centroids
      J = np.where(classifications == j)
      X_C = X[J]
      new_centroids[j] = X_C.mean(axis = 0)

      # Compute loss
      for i in range(0, X_C.shape[0]):
        new_loss += np.sum(np.power(X_C[i, :] - centroids[j], 2))

    # Stopping criterion            
    if np.abs(loss - new_loss) < tolerance:
    #if (new_centroids == centroids).all():
      return new_centroids, classifications, new_loss
    
    centroids = new_centroids
    loss = new_loss

  print("Failed to converge!")
  return centroids, classifications, loss

centers, classifications, loss = kmeans(X, 4)

# Plot
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()
s1 = plt.subplot(1, 2, 1)
s1.set_title("Estimated clusters")
s1.scatter(X[:, 0], X[:, 1], c = classifications, s = 2)
s1.scatter(centers[:, 0], centers[:,1], c = "r", s = 20)
s2 = plt.subplot(1, 2, 2)
s2.set_title("True clusters")
s2.scatter(X[:, 0], X[:, 1], c = y, s = 2)
# Generate html image
plt.savefig('tmp2.png', bbox_inches='tight')
import base64
data_uri = base64.b64encode(open('tmp2.png', 'rb').read()).decode('utf-8').replace('\n', '')
img_tag = '<img height="300" src="data:image/png;base64,%s">' % data_uri
with open("tmp2.txt", "w") as text_file:
    text_file.write(img_tag)

# Choosing cluster size
# Not very good actually.
K = np.arange(2, 8)
losses = np.zeros(8 - 2)
for i in range(0, 8 - 2):
    centers, classifications, loss = kmeans(X, K[i])
    losses[i] = loss

fig = plt.figure(figsize=(10, 6))
fig.tight_layout()
plt.plot(K, losses, marker = "o")
plt.xlabel("k")
plt.ylabel("loss")
# Generate html image
plt.savefig('tmp3.png', bbox_inches='tight')
import base64
data_uri = base64.b64encode(open('tmp3.png', 'rb').read()).decode('utf-8').replace('\n', '')
img_tag = '<img height="300" src="data:image/png;base64,%s">' % data_uri
with open("tmp3.txt", "w") as text_file:
    text_file.write(img_tag)
