# Interactive Twitter Bot Detection through Network Graph Analysis

## Project setup 

- Prereq: You have anaconda/miniconda installed and set up on your machine

- Create the environment from the given environment.yml file and activate it
    
    ```bash
    conda env create -f environment.yml
    conda activate tbd
    ```

- Start streamlit
        
    ```bash
    streamlit run streamlit/main.py
    ```

## Steps to use the app

- In bot_detection.py, fill in your Twitter API credentials. This is needed for getting the user handle from user id. We need the user handle to display the profile on the app. Also make sure you have tweepy installed. 

- Load the clusters into the DB first. The UI is now picking up cluster and embedding information from the sqlite DB. clusters.pkl is the pickle file containing the clusters and userids (given by John). The script will load the clusters into the DB. The --reset flag is used to reset the DB. If you want to load the clusters into an existing DB, then set reset to 0.

    ```bash
    cd scripts
    python load_clusters.py --file clusters.pkl --reset 1
    ```

- Run the streamlit app

    ```bash
    streamlit run streamlit/main.py
    ```
