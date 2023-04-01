import argparse
import os
import sys

import numpy as np
import pandas as pd

# append the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db_utils import init_database, load_clusters


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # get the pkl file from the command line
    parser.add_argument('--file', type=str, help='Path to the pkl file')

    # check if database needs to be reset
    parser.add_argument('--reset', type=int, help='Reset the database 1->yes, 0->no')

    # parse the arguments
    args = parser.parse_args()

    # get the pkl file path
    pkl_file = args.file

    # check if database needs to be reset
    reset = args.reset

    if reset:
        init_database(reset=True)

    # load the clusters
    if load_clusters(pkl_file):
        print('Successfully loaded clusters')
    else:
        print('Failed to load clusters')