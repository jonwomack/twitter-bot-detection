import os
import sys

import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# append the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.graph import Graph


def run_ui():

    # define columns for the nodes and edges csv upload widgets
    col1, col2 = st.columns([1,1])

    # nodes.csv upload widget
    with col1:
        nodes_csv = st.file_uploader('nodes.csv', type='csv', label_visibility="hidden")

    # edges.csv upload widget
    with col2:
        edges_csv = st.file_uploader('edges.csv', type='csv', label_visibility="hidden")

    # create generate button
    generate_button = st.button('Generate Graph')

    # if generate button is clicked
    if generate_button:

        # if nodes.csv is not uploaded
        if nodes_csv is None:
            st.error('Please upload nodes.csv')
            return

        # if edges.csv is not uploaded
        if edges_csv is None:
            st.error('Please upload edges.csv')
            return

        # read nodes.csv
        nodes_df = pd.read_csv(nodes_csv)

        # read edges.csv
        edges_df = pd.read_csv(edges_csv)

        # init graph
        graph = Graph()

        # construct graph from nodes and edges dataframes
        graph.construct_graph_from_dataframe(nodes_df, edges_df)

        # visualize the graph
        html = graph.visualize_graph()

        # display the graph
        components.html(html, height=800)

