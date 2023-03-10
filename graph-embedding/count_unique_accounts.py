import csv

unique_accounts = []
with open('graphwave-database-6000.csv', 'r') as file:
    lines = file.readlines()
    for row in lines:
        entry1, entry2 = row.split(',')
        entry1 = entry1.replace("\n", "")
        entry2 = entry2.replace("\n", "")
        unique_accounts.append(entry1)
        unique_accounts.append(entry2)
print(unique_accounts)
print(len(set(unique_accounts)))
