# -*- coding: utf-8 -*-
"""
April 2023
Code by Madelyn Scandlen
"""

import pandas as pd
import numpy as np

df = pd.read_csv('./out/Twibot-20_predictions.csv')

df = df.groupby(['id'])['probability'].mean()
df = df.reset_index()
df['id'] = df['id'].apply(lambda x: 'u' + str(x))
print(df.head())

df.to_csv('./out/Twibot-20_user_predictions.csv', index=False)
