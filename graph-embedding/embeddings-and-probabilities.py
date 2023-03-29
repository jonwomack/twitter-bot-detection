import argparse
import csv
import json
import os
import pickle
import random
from matplotlib import pyplot as plt



# Get mapping from twitter userid to embedding index
f = open('userid_to_index.json')
userid_to_index = json.load(f)
keys = list(userid_to_index.keys())
values = list(userid_to_index.values())

# Get ground truth labels for twitter userids
userid_to_ground_truth = {}
with open("../datasets/twibot-20/label.csv", 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader, None) # skip first row with column titles
    for row in csvreader:
        userid_to_ground_truth[row[0]] = row[1]



def trunc_gauss(mu, sigma, bottom, top):
    a = random.gauss(mu,sigma)
    while (bottom <= a <= top) == False:
        a = random.gauss(mu,sigma)
    return a

def generate_probabilitity_based_on_ground_truth(ground_truth):
    if ground_truth == 'human':
        probability = trunc_gauss(.75, .15, 0, 1)
    if ground_truth == 'bot':
        probability = trunc_gauss(.25, .15, 0, 1)
    return probability

def generate_probability_based_on_ground_truth_binary(ground_truth):
    classifier_accuracy = .72
    if ground_truth == 'human':
        if random.random() < 1 - classifier_accuracy:
            # mis-classify human as bot 20% of the time i.e. 
            classification = 0
        else:
            classification = 1
    if ground_truth == 'bot':
        if random.random() < 1 - classifier_accuracy:
            # mis-classify bot as human 20% of the time i.e. 
            classification = 1
        else:
            classification = 0
    return classification
    


def saveEmbeddings(embeddings_file):
    num_labels = 0
    id_embedding_probability = []
    humans = []
    bots = []
    with open(embeddings_file, 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader, None) # skip first row with column titles
        for row in csvreader:
            # print(row)
            # print(row[0])
            # print("here")
            key_index_for_userid = values.index(int(row[0]))

            # User ID for visualizing account
            user_id = keys[key_index_for_userid]

            # Embedding for clustering
            embedding_excluding_idx = row[1::]

            # Probability for clustering
            try:
                ground_truth = userid_to_ground_truth[user_id]
                probability = generate_probabilitity_based_on_ground_truth(ground_truth)
                if ground_truth == 'bot':
                    ground_truth_value = 0
                    bots.append(probability)
                elif ground_truth == 'human':
                    humans.append(probability)
                    ground_truth_value = 1
                id_embedding_probability.append([user_id, embedding_excluding_idx, probability, ground_truth_value])
                num_labels += 1
            except:
                continue


            # print(row)

    id_embedding_probability_file = os.path.splitext(embeddings_file)[0] + "-id-emb-prob.pkl"

    with open(id_embedding_probability_file, 'wb') as file:
        pickle.dump(id_embedding_probability, file)

    ## Plot of bots and humans probabilities
    # fig, ax = plt.subplots(figsize =(10, 7))
    # ax.hist([bots, humans], bins = 20, color=['red', 'blue'])
    # # Show plot
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-c", type=str, help="P")
    args = parser.parse_args()
    embeddings_path = os.path.basename(args.path)
    print(embeddings_path)
    saveEmbeddings(embeddings_path)