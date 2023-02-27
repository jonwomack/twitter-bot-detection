


import csv
from collections import defaultdict


users = []
user_follows = defaultdict(list)
with open("/Users/jonathanwomack/projects/subgraph-mining/Twibot-20/follow-friends-edge.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    users.append(row[0])
    users.append(row[2])
    user_follows[row[0]].append(row[2])
print(len(users))

list_of_users = list(set(users))

userid_to_index = {}
with open('database.txt', 'a') as outfile:
    outfile.write('# t 1\n')
    for index, userid in enumerate(list_of_users):
      current_vertex = "v " + str(index) + " 1\n"
      userid_to_index[userid] = index
      outfile.write(current_vertex)

    
    for user, list_of_following in user_follows.items():
      for following in list_of_following:
        user_index = userid_to_index[user]
        following_index = userid_to_index[following]
        current_edge = "e " + str(user_index) + " " + str(following_index) + " 1\n"
        outfile.write(current_edge)
  

# for i in len(edges):
