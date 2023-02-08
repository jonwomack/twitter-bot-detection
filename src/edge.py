import os
import sys

import numpy as np
import pandas as pd


class Edge:

    # TODO: Think of more attributes to add to the edge
    def __init__(self, source, target, weight):
        """
        Edge constructor
        Args:
            source (int): source node id
            target (int): target node id
            weight (float): edge weight
        """

        self.source = source
        self.target = target
        self.weight = weight
