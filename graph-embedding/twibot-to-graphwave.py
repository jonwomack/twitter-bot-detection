import json
import csv
from collections import defaultdict


users = []
user_follows = defaultdict(list)
with open("datasets/twibot-20/follow-friends-edge.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    users.append(row[0])
    users.append(row[2])
    user_follows[row[0]].append(row[2])
print(len(users))

list_of_users = list(set(users))

userid_to_index = {}
with open('graphwave-database-twibot20.txt', 'a') as outfile:
    for index, userid in enumerate(list_of_users):
      userid_to_index[userid] = index

    outfile.write("node_1,node_2\n")
    for user, list_of_following in user_follows.items():
      for following in list_of_following:
        user_index = userid_to_index[user]
        following_index = userid_to_index[following]
        current_edge = str(user_index) + "," + str(following_index) + "\n"
        outfile.write(current_edge)

with open('userid_to_index.json', 'w') as convert_file:
    convert_file.write(json.dumps(userid_to_index))
