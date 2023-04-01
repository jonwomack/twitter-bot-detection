import os
import sys

import pickle
import pandas as pd
import requests
from sklearn.manifold import TSNE

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

import tweepy

from src.db_utils import get_unlabeled_clusters, get_cluster_embeddings, get_cluster_userids

# twitter auth for getting user handles
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# append the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.graph import Graph

@st.cache_data
def embed_tweet(twitter_user_id):

    # use the twitter api to get the user handle
    user = api.get_user(user_id=twitter_user_id)
    handle = user.screen_name
    profile_url = f"https://twitter.com/{handle}"
    embed_api = f"https://publish.twitter.com/oembed?url={profile_url}"

    st.markdown(f"### Twitter Profile for handle: {handle}")

    # get the html from the api
    response = requests.get(embed_api)

    # extract the html from the response
    html = response.json()['html']

    return html

@st.cache_data
def generate_tsne_graph(cluster):
    """
    This function performs a tsne reduction for tje given cluter
    Args:
        cluster (list): 
            - list with 2 elements.
            - first element is a np array of embeddings
            - second element is a list of user ids
    Returns:
        html (str): html string
    """

    embeddings, userids = cluster

    # first, initialize a t-SNE object with the desired hyperparameters
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)

    embeddings_2d = tsne.fit_transform(embeddings)

    # create a dataframe with the 2d embeddings
    df_embeddings_2d = pd.DataFrame(embeddings_2d, columns=['x', 'y'])

    # add column for user ids
    df_embeddings_2d['user_id'] = userids

    # create a scatter plot
    fig = px.scatter(df_embeddings_2d, x='x', y='y', width=600, height=600, hover_data=['user_id'])

    # add a title to the plot
    fig.update_layout(title_text='t-SNE Visualization of User Embeddings for given cluster')

    # convert plotly figure to html
    html = fig.to_html()

    return html

def run_ui():

    # define columns
    col1, col2 = st.columns([2,1])

    # visualization section
    with col1:

        # create a form to get user input
        with st.form('graph'):

            # get all the unlabeled clusters
            clusters = get_unlabeled_clusters()

            # create a dropdown to select which cluster to visualize
            cluster_id = st.selectbox('Select a cluster', options=clusters)

            # create generate button
            generate_button = st.form_submit_button('Generate Graph')

            # if generate button is clicked
            if generate_button:
                
                # centering the graph
                _, col5, _ = st.columns([1,2,1])
                with col5:

                    with st.spinner('Generating Graph...'):

                        cluster_embeds = get_cluster_embeddings(cluster_id)
                        cluster_userids = get_cluster_userids(cluster_id)

                        # generate graph for the selected cluster
                        html = generate_tsne_graph([cluster_embeds, cluster_userids])
                        components.html(html, width=600, height=600)

    # human in loop section
    with col2:

        # create a form to get user input
        with st.form('analyse'):

            st.markdown("## Analyse")

            # if generate button is clicked
            if generate_button:
                
                # get the twitter id to search
                twitter_user_id = st.selectbox('Search by ID', options=cluster_userids)

                # # load the twitter profile as an iframe when the user selects a twitter id
                with st.spinner('Loading Profile...'):
                    components.html(embed_tweet(twitter_user_id), width=600, height=600, scrolling=True)

                # user input to determine if bot or not
                is_bot = st.radio('Is this a bot?', options=['Yes', 'No'])

            # submit button
            submit_button = st.form_submit_button('Submit')
