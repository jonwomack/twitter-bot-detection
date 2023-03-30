import argparse
import os
import pickle
import numpy as np
from sklearn.datasets import make_blobs
from fastcluster import linkage
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


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

def getHeterogenity(items, ground_truths):
   cluster_ground_truths = [ground_truths[i] for i in items]
   return sum(cluster_ground_truths)/len(cluster_ground_truths)

def getProbability(items, probabilties):
    simulated_cluster_probabilities = [probabilties[i] for i in items]
    return sum(simulated_cluster_probabilities)/len(simulated_cluster_probabilities)




def plotHeterogenity(id_embedding_probability_file):
    graph_embedding_path = "../graph-embedding/"
    # Open the file in binary mode

    user_id = []
    probabilities = []
    np_embeddings = []
    ground_truths = []
    selected_clusters = []
    with open(graph_embedding_path + id_embedding_probability_file, 'rb') as file:
        
        # Call load method to deserialze
        id_embedding_probability = pickle.load(file)
        for user in id_embedding_probability:
            user_id.append(user[0])
            string_embedding = np.asarray(user[1])
            np_embedding = string_embedding.astype(np.float64)
            np_embeddings.append(np_embedding)
            probabilities.append(user[2])
            ground_truths.append(user[3])
        print(len(probabilities))
        print(len(ground_truths))
        print(len(np_embeddings))
    # print(user_id[0])
    # print(np_embeddings[0])
    # print(probabilities[0])
    # print(ground_truths)
    np_embeddings = np.asarray(np_embeddings)
    # Generate synthetic dataset with n points in 100 dimensions and 3 blobs
    # X, y = make_blobs(n_samples=n, n_features=100, centers=3, random_state=42)
    # print(y)
    # # print(X)
    # print(type(X))
    X = np_embeddings
    y = ground_truths
    # Use fastcluster to cluster the data
    Z = linkage(X, method='ward')

    #start at the super node
    supernodeInd = getRootClusterIndex(Z)

    #investigate the items in the supernode
    items = getClusterItemInds(supernodeInd, Z)
    # items are 0-indexed, root node has all embeddings

    #get ids of its two sub-clusters
    clustA, clustB = getSubClusters(supernodeInd, Z)


    remaining_clusters = [supernodeInd]

    cluster_sizes = []
    cluster_heterogenities = []
    while len(remaining_clusters) > 0:
        current_cluster_idx = remaining_clusters.pop(0)
        current_cluster_items = getClusterItemInds(current_cluster_idx, Z)    
        # heterogenity = getHeterogenity(current_cluster_items, ground_truths)
        heterogenity = getProbability(current_cluster_items, probabilities)
        # if heterogenity < 1:
        if len(current_cluster_items) < 100:
            cluster_sizes.append(len(current_cluster_items))
            cluster_heterogenities.append(heterogenity)
            if heterogenity < .5:
                selected_embeddings = [np_embeddings[i] for i in current_cluster_items]
                selected_userids = [user_id[i] for i in current_cluster_items]
                # print(selected_embeddings)
                # print(selected_userids)
                selected_clusters.append([selected_embeddings, selected_userids])
        try:
            clustA, clustB = getSubClusters(current_cluster_idx, Z)
            remaining_clusters.append(clustA)
            remaining_clusters.append(clustB)
        except:
            continue
    selected_clusters_sorted = sorted(selected_clusters, key=lambda x: len(x[0]), reverse=True)    
    selected_clusters_sorted_file = id_embedding_probability_file.split("-", 1)[0] + "-ui.pkl"
    with open(selected_clusters_sorted_file, 'wb') as file:
        pickle.dump(selected_clusters_sorted, file)

    # To select embedding: want high uniformity

    # Starting with ground truth probabilities (i.e. oracle nlp), then move on to erroneous probs


    # Plotting
    # plt.scatter(cluster_sizes, cluster_heterogenities)
    # plt.savefig(os.path.splitext(id_embedding_probability_file)[0] + '.png')

    # Handoff to UI






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-c", type=str, default="approx4sample4-id-emb-prob.pkl", help="P")
    args = parser.parse_args()
    embeddings_path = os.path.basename(args.path)
    print(embeddings_path)
    plotHeterogenity(embeddings_path)
    print()