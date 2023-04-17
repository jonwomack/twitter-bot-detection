# Interactive Twitter Bot Detection through Network Graph Analysis

## Project setup 

- Prereq: You have anaconda/miniconda installed and set up on your machine

- Create the environment from the given environment.yml file and activate it
    
    ```bash
    conda env create -f environment.yml
    conda activate tbd
    ```
- Install a few more dependencies

    ```
    pip install scikit-learn plotly tweepy streamlit==1.20.0
    ```

- Place the bearer token string in ```streamlit/token.txt```. This is used to get twitter handles for embedding the profile. 


- Start streamlit
        
    ```bash
    streamlit run streamlit/main.py
    ```

## Steps to use the app

- Load the clusters into the DB first using the given script. The UI is now picking up cluster and embedding information from the sqlite DB. The --reset flag is used to reset the DB. If you want to load new clusters into an existing DB, then set reset to 0.

    ```bash
    cd scripts
    python load_clusters.py --file {pkl_file} --reset 1
    ```

- Run the streamlit app

    ```bash
    streamlit run streamlit/main.py
    ```

- The sqlite DB is generated in the root directory. You can query the DB to get the user information. 

    ```bash
    sqlite3 tbd.db
    ```

    For example, to get the first 10 users, run the following commands in the shell

    ```sql
    .mode column
    .headers on
    select * from users limit 10;
    ```