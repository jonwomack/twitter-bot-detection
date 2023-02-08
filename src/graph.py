import os
import sys

import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx

from src.node import Node
from src.edge import Edge


class Graph:

    def __init__(self):
        """
        Graph constructor
        """
        
        # initialize the graph
        self.graph = nx.Graph()

        # node mapper
        self.node_mapper = {}

        # edge mapper
        self.edge_mapper = {}

    def construct_graph_from_dataframe(self, nodes_df, edges_df):
        """
        Constructs a graph from the nodes and edges dataframes
        Args:
            nodes_df (pd.DataFrame): nodes dataframe
            edges_df (pd.DataFrame): edges dataframe
        """

        # iterate over the nodes dataframe and add nodes to the graph
        for index, row in nodes_df.iterrows():
            
            # extract node attributes
            node_id = int(row['node_id'])
            twitter_user_id = int(row['twitter_user_id'])
            bot_probability = row['bot_probability']
            twitter_username = row['twitter_username']

            # higher the bot probability, darker the red shade
            node_color = f'rgba({255 - int(bot_probability * 255)}, 0, {int(bot_probability * 255)}, 1)'

            # create a node object
            node = Node(node_id, twitter_user_id, bot_probability, twitter_username, node_color)

            # add node to the graph
            self.graph.add_node(node_id, label=twitter_user_id, title=f'user_id: {twitter_user_id}', color=node_color)

            # add node to the node mapper
            self.node_mapper[node_id] = node

        # iterate over the edges dataframe and add edges to the graph
        for index, row in edges_df.iterrows():
            
            # extract edge attributes
            edge_id = int(row['edge_id'])
            source = int(row['source'])
            target = int(row['target'])
            weight = float(row['weight'])

            # create an edge object
            edge = Edge(source, target, weight)

            # add edge to the graph
            self.graph.add_edge(source, target, weight=weight, title=weight, color='rgba(0, 0, 0, 1)', width=0.1)

            # add edge to the edge mapper
            self.edge_mapper[edge_id] = edge

    def visualize_graph(self, return_net=False, return_html=True):
        """
        Visualizes the graph with pyvis
        Args:
            return_net (bool): whether to return the pyvis network object
            return_html (bool): whether to return the html string
        Returns:
            net (pyvis.network.Network): pyvis network object
            or html (str): html string
        """

        # initialize the network
        net = Network()

        # build net from networkx graph
        net.from_nx(self.graph)

        if return_net:
            return net
        
        elif return_html:
            return net.generate_html(name='./html/test.html', notebook=False)



    