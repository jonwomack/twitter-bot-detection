import os
import sys

import numpy as np
import pandas as pd


class Node:

    # TODO: Think of more attributes to add to the node
    def __init__(self, 
                node_id, 
                twitter_user_id, 
                bot_probability, 
                twitter_username,
                node_color):
        """
        Node constructor
        Args:
            node_id (int): node id
            twitter_user_id (int): twitter user id
            bot_probability (float): bot probability
            twitter_username (str): twitter username
        """

        self.node_id = node_id
        self.twitter_user_id = twitter_user_id
        self.bot_probability = bot_probability
        self.twitter_username = twitter_username
        self.node_color = node_color