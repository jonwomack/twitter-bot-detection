 # -*- coding: utf-8 -*-
"""
April 2023
Updated code by Madelyn Scandlen
Original code by Shangbin Feng et al. found at https://github.com/LuoUndergradXJTU/TwiBot-22/blob/master/src/Wei/Twibot-20/data_processor.py
Inspired by Feng Wei & Uyen Trang Nguyen
user feature generation: Alhosseini
"""

import re
import gc
import datetime
import pandas as pd
import numpy as np
import ijson
from tqdm import tqdm

def clean_text(input):
    lowercase = input.lower()
    url_token = re.sub('https://t.co\/[^\s]+', ' <url> ', lowercase)
    hashtag_token = re.sub('#[^\s]+', ' <hashtag> ', url_token)
    mention_token = re.sub('@[^\s]+', ' <user> ', hashtag_token)
    retweet_token = re.sub(r'rt\s+', ' rt ', mention_token)
    strip_punc = re.sub('[^a-zA-Z0-9\<\>]', ' ', retweet_token)
    strip_num = re.sub('(?<=\\d) +(?=\\d)', '', strip_punc)
    num = re.sub('\d+', ' <number> ', strip_num)
    strip_nl = re.sub(r'\s\s+', ' ', num)
    return strip_nl

path = "./datasets/Twibot-20/"
dl = pd.read_csv(path+"label.csv")
ds = pd.read_csv(path+"split.csv")

ds = ds[ds.split != "support"]
ds = pd.merge(ds, dl,  left_on='id', right_on='id')

de = pd.read_csv(path+'edge.csv')
de = de[de.relation == "post"]
de = de[de.source_id.isin(ds.id) ]

dsde = pd.merge(ds, de,  left_on='id', right_on='source_id')
del dsde["source_id"]

ids = []
texts = []

with open(path+'node.json', 'rb') as f:
    for node in tqdm(ijson.items(f, 'item')):
        if node['id'][0] == 't':
            ids.append(node['id'])
            texts.append(clean_text(node['text']))
df = pd.DataFrame(list(zip(ids, texts)), columns = ['id', 'text'])
out=pd.merge(dsde, df,  left_on='target_id', right_on='id')
del dsde, df, ids, texts
out.dropna(inplace = True)
out['id'] = out['id_x'].str[1:].astype('int')
out = out.drop(columns=['id_x', 'id_y'])

gc.collect()

print("Data")
print(out.head())

out.to_json("./datasets/Twibot-20/Twibot-20.json", orient='records')
