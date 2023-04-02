# user feature generation: Alhosseini

import json
import datetime
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

path = "./datasets/Twibot-20/"
dl = pd.read_csv(path+"label.csv")
ds = pd.read_csv(path+"split.csv")
print("Load Data")
ds = ds[ds.split != "support"]
ds = pd.merge(ds, dl,  left_on='id', right_on='id')

de = pd.read_csv(path+'edge.csv')
de = de[de.relation == "post"]
de = de[de.source_id.isin(ds.id) ]

print("Label-Split")

dsde = pd.merge(ds, de,  left_on='id', right_on='source_id')
del dsde["source_id"]

print("Nodes")

for chunk in tqdm(pd.read_json(path+"node.json", lines=True, chunksize = 100)):
    print("Chunk ", i)
    data=chunk[['id','text']]
    out=pd.merge(dsde, data,  left_on='target_id', right_on='id')
    out.dropna(inplace = True)
    out.to_json("./Twibot-20" + "-" + i + ".json")
