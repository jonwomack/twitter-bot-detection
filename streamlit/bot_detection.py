import os
import sys

import pandas as pd
from sklearn.manifold import TSNE

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

# append the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.graph import Graph

@st.cache_data
def generate_tsne_graph(df_embeddings):
    """
    This function performs a tsne reduction on the embeddings and returns a html of the scatter plot.
    Args:
        df_embeddings (pd.DataFrame): embeddings dataframe
    Returns:
        html (str): html string
    """
    
    # convert embeddings dataframe to numpy array
    embeddings = df_embeddings.values

    # first, initialize a t-SNE object with the desired hyperparameters
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)

    embeddings_2d = tsne.fit_transform(embeddings)

    # create a dataframe with the 2d embeddings
    df_embeddings_2d = pd.DataFrame(embeddings_2d, columns=['x', 'y'])

    # create a scatter plot
    fig = px.scatter(df_embeddings_2d, x='x', y='y', width=800, height=800)

    # add a title to the plot
    fig.update_layout(title_text='t-SNE Visualization of User Embeddings')

    # TODO:
    # 1. change the color of points based on their bot probability
    # 2. add hover text to the points containing details of the user

    # convert plotly figure to html
    html = fig.to_html()

    return html

def run_ui():
    #st.title('BOT DETECTION')

    # define columns
    col1, col2 = st.columns([2,1])

    # embeddings.csv upload widget
    with col1:
        st.markdown("## Upload the embeddings file")
        embeddings_csv = st.file_uploader('embeddings.csv', type='csv', label_visibility="hidden")

        # create generate button
        generate_button = st.button('Generate Graph')

    # human in loop section
    with col2:
        st.markdown("## Analyse")
        
        # take a number input for user id
        # TODO:
        # 1. Add dropdown for all the user ids
        # 2. Add feature to search by name

        # add nested columns to get user input
        col3, col4 = st.columns([1,1])

        with col3:
            search_by_id = st.text_input('Search by id')

        with col4:
            search_by_name = st.text_input('Search by name')
        
        # create a button to search
        search_button = st.button('Search')
        

    # if generate button is clicked
    if generate_button:

        # if embeddings.csv is not uploaded
        if embeddings_csv is None:
            st.error('Please upload embeddings.csv')
            return

        # read embeddings.csv
        df_embeddings = pd.read_csv(embeddings_csv)

        # generate graph
        html = generate_tsne_graph(df_embeddings)

        # visualize the graph
        components.html(html, height=800)