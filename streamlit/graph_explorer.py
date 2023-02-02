import os
import sys

import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

def run_ui():
    st.title('GRAPH EXPLORER')

    net = Network()

    net.add_node(1, label='Node 1', title='Node 1')
    net.add_node(2, label='Node 2', title='Node 2')
    net.add_edge(1, 2, value=1, title='Follower')

    net.save_graph('./html/test.html')

    html_file = open('./html/test.html', 'r', encoding='utf-8')
    components.html(html_file.read(), height=800)