import os
import sys

import numpy as np
import pandas as pd

import streamlit as st

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# streamlit pages
import graph_explorer, bot_detection

# TODO: Add more pages later
PAGES = [
    'GRAPH EXPLORER',
    'BOT DETECTION',
]

def run_ui():

    st.set_page_config(
        page_title="Twitter Bot Analysis",
        page_icon="🏠",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    st.sidebar.title("Twitter Bot Analysis")
    if st.session_state.page:
        page=st.sidebar.radio('Navigation', PAGES, index=st.session_state.page)
    else:
        page=st.sidebar.radio('Navigation', PAGES, index=0)

    # TODO: Add more pages later
    if page == 'GRAPH EXPLORER':
        st.sidebar.write("""
            ## Overview
            This page presents an interactive graph explorer for the Twitter user network.
        """)

        graph_explorer.run_ui()

    elif page == 'BOT DETECTION':

        st.sidebar.write("""
            ## Overview
            This page presents a bot detection model for the Twitter user network.
        """)

        bot_detection.run_ui()

if __name__ == "__main__":
    
    # show the graph explorer page by default
    st.session_state.page = 0
    
    run_ui()