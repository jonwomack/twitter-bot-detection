import numpy as np
from sklearn.datasets import make_blobs
from fastcluster import linkage
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


n = 100

# Generate synthetic dataset with n points in 100 dimensions and 3 blobs
X, y = make_blobs(n_samples=n, n_features=100, centers=3, random_state=42)

# Use fastcluster to cluster the data
Z = linkage(X, method='ward')


def getClusterItemInds(index, dendo = None):
  """
  Returns an array of item indicies of a given cluster index. If index is less
  than n, then it is a cluster of itself (singleton cluster). 
  
  This function runs recursively.

  The root cluster

  Args:
  index (int): the index of the cluster to retrieve the item indicies from
  dendo (numpy.ndarray): the stepwise dendogram output of a linkage() call

  Returns:
  array: an array of item indicies

  """

  # n is the number of items in the dataset (clustering input)
  n = len(dendo) + 1

  if index < 0 or index > 2*n-2:
    raise Exception("index out of bounds")

  if index < n:
    return [int(index)]

  dendoInd = int(index - n)
  
  indices = []
  
  dendoStep = dendo[dendoInd]
  indA, indB, dist, count = dendoStep

  indices.extend(getClusterItemInds(indA, dendo))
  indices.extend(getClusterItemInds(indB, dendo))
  
  return indices


def getRootClusterIndex(dendo = None):
  """
  Returns the cluster index of the one big node

  Args:
  dendo (numpy.ndarray): the stepwise dendogram output of a linkage() call

  Returns:
  int: cluster index of the super cluster

  """
  n = len(dendo) + 1
  return 2*n-2

def getSubClusters(index, dendo = None):
  """
  Given a cluster index, return the indicies of its two sub clusters

  Args:
  index (int): the index of the cluster to retrieve its sub cluster indicies
  dendo (numpy.ndarray): the stepwise dendogram output of a linkage() call

  Returns:
  tuple (int, int): tuple of two cluster indicies

  """
  n = len(dendo) + 1

  if index < 0 or index > 2*n-2:
    raise Exception("index out of bounds")

  if index < n:
    raise Exception("index is of a singleton cluster")

  dendoInd = int(index - n)

  dendoStep = dendo[dendoInd]
  indA, indB, _, _ = dendoStep

  return (int(indA), int(indB))


#example workflow

#start at the super node
supernodeInd = getRootClusterIndex(Z)

#investigate the items in the supernode
items = getClusterItemInds(supernodeInd, Z)

#use items to generate visualization
embeddings = [X[items] for ind in items]

#get ids of its two sub-clusters
clustA, clustB = getSubClusters(supernodeInd, Z)

#select sub-cluster clustA items to investigate
itemsA = getClusterItemInds(clustA, Z)

#use itemsA to generate visualization

#repeat

dendro = dendrogram(Z)

# Visualize the dendrogram
plt.title('Dendrogram')
plt.ylabel('Distance')
plt.show()


# Visualize the clusters in 2D using t-SNE
X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)


# Plot the results in 2d t-sne
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
legend = plt.legend(*scatter.legend_elements(), title="Categories", loc='upper right', bbox_to_anchor=(1.22, 1))
plt.show()