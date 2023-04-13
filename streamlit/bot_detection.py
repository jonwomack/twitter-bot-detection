import os
import sys
import time

import pickle
import pandas as pd
import requests
from sklearn.manifold import TSNE

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

import tweepy

from src.db_utils import get_unlabeled_clusters, get_cluster_embeddings, get_cluster_userids, get_user, label_users

# TWITTER API CONFIG

# MAKE SURE TO DELETE THIS BEFORE PUSHING TO GITHUB
bearer_token = ""
params = {
    "user.fields": "username"
}
headers = {
    "Authorization": f"Bearer {bearer_token}",
    "User-Agent": "v2UserLookupPython"
}

def get_username_by_id(twitter_user_id):
    """
    This function gets the username by twitter user id
    Args:
        twitter_user_id (int): twitter user id
    Returns:
        username (str): twitter username
    """
    url = f"https://api.twitter.com/2/users/{twitter_user_id}"
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    
    username = response.json()['data']['username']
    return username

# append the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.graph import Graph

# global dataframe to store annotation information before submitting
df = pd.DataFrame(columns=['user_id', 'cluster_id', 'label'])

@st.cache_data
def embed_tweet(twitter_user_id):

    handle = get_username_by_id(twitter_user_id)
    profile_url = f"https://twitter.com/{handle}"
    embed_api = f"https://publish.twitter.com/oembed?url={profile_url}"

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
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, n_iter=1000)

    embeddings_2d = tsne.fit_transform(embeddings)

    # create a dataframe with the 2d embeddings
    df_embeddings_2d = pd.DataFrame(embeddings_2d, columns=['x', 'y'])

    # add column for user ids
    df_embeddings_2d['user_id'] = userids

    # create a scatter plot
    fig = px.scatter(df_embeddings_2d, x='x', y='y', width=800, height=800, hover_data=['user_id'])

    # add a title to the plot
    fig.update_layout(title_text='t-SNE Visualization for the given cluster')

    # convert plotly figure to html
    html = fig.to_html()

    return html

def run_ui():

    # define columns
    col1, col2 = st.columns([2,1])

    # visualization section
    with col1:


        # get all the unlabeled clusters
        clusters = get_unlabeled_clusters()

        # create a dropdown to select which cluster to visualize
        cluster_id = st.selectbox('SELECT A CLUSTER TO ANNOTATE', options=clusters)

        # add line spacing
        st.markdown('---')

        # centering the graph
        _, col3, _ = st.columns([1,3,1])
        with col3:

            with st.spinner('Generating Graph...'):

                cluster_embeds = get_cluster_embeddings(cluster_id)
                cluster_userids = get_cluster_userids(cluster_id)

                # generate graph for the selected cluster
                html = generate_tsne_graph([cluster_embeds, cluster_userids])
                components.html(html, width=800, height=800)


    # human in loop section
    with col2:

        # get the twitter id to search
        twitter_user_id = st.selectbox('SEARCH BY ID', options=cluster_userids)

        # add line spacing
        st.markdown('---')

        # add this twitter id to the session state
        st.session_state['twitter_user_id'] = twitter_user_id

        # create a form to get user input
        with st.form('analyse'):

            # # load the twitter profile as an iframe when the user selects a twitter id
            with st.spinner('Loading Profile...'):

                twitter_user_id = st.session_state['twitter_user_id']
                components.html(embed_tweet(twitter_user_id), height=500, scrolling=True)

                # get user information from db and display
                user = get_user(twitter_user_id)

                user_id = user[0]
                cluster_id = user[1]
                label = user[2]

                # display information as a table
                _, col6, _ = st.columns([1,3,1])
                with col6:
                    st.dataframe(pd.DataFrame({'TWITTER ID': [user_id], 'CLUSTER ID': [cluster_id], 'LABEL': [label]}), use_container_width=True)

                # annotation section
                _, col7, _ = st.columns([1,1,1])
                with col7:
                    # user input to determine if bot or not
                    label = st.radio('Label', options=['BOT', 'NOT BOT'], horizontal=True, label_visibility='collapsed')
                
                    # submit button
                    mark_button = st.form_submit_button('MARK', use_container_width=True)

            # if the user clicks submit, add the information to the dataframe
            if mark_button:

                # check if this row is already in the dataframe
                if df[df['user_id'] == twitter_user_id].empty:
                    # add the information to the dataframe
                    df.loc[len(df)] = [twitter_user_id, cluster_id, 2 if label == 'BOT' else 1]
                else:
                    # update the information in the dataframe
                    df.loc[df['user_id'] == twitter_user_id, 'label'] = 2 if label == 'BOT' else 1

    # show the annotations done so far
    st.markdown('---')
    
    _, col4, _ = st.columns([1,3,1])
    with col4:
        st.dataframe(df, use_container_width=True)

        _, col5, _ = st.columns([1,3,1])
        with col5:
            # submit button to save the annotations to the database
            submit_button = st.button('SAVE ANNOTATIONS TO DATABASE', use_container_width=True)

            # if the user clicks submit, save the annotations to the database
            if submit_button:

                # save the annotations to the database
                label_users(df)

                # clear the dataframe
                df.drop(df.index, inplace=True)

                # clear the session state
                st.session_state.clear()

                # show a success message
                st.success('Annotations saved successfully!')

                # wait for 3 seconds and then reload the page
                time.sleep(3)
                st.experimental_rerun()

