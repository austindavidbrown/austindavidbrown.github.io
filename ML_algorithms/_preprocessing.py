import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/iris.txt", header = None)
df.head().to_html("iris.html")

df.info()

df.describe()

df.hist(figsize = (10, 10))
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.8, figsize=(10, 10), diagonal='kde')
plt.show()


