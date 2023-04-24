# Interactive Twitter Bot Detection through Network Graph Analysis

## DESCRIPTION

- This is a package that provides a Streamlit-based user interface for annotating Twitter user profiles as bots or not. It leverages SQLite databases to store clusters and labels, making it easy to manage the annotation process. The annotator can perform t-SNE reductions to visualize clusters and examine multiple user profiles within the same cluster. By labeling multiple profiles and submitting, a majority vote determines the cluster's label, significantly accelerating the manual labeling process. Streamlit's native caching capabilities and ease of use make it an ideal choice for this application.

- The clusters were generated using the Fastcluster Python library, which executes hierarchical clustering on high-dimensional embedding data with Ward's method. This technique minimizes intra-cluster variance, producing balanced and compact clusters. The library outputs a dendrogram matrix, and utility functions were developed to select and investigate sub-clusters based on their cluster scores. This enables a comprehensive understanding of cluster patterns and relationships, creating a useful tool for annotating bot clusters. The method completes hierarchical clustering on 11k embeddings in under 10 minutes, and for larger datasets like Twibot-22, the dataset will be partitioned to maximize propagated labels.

- Behind the scenes, we employ natural language processing techniques on user tweets from the Twibot-20 dataset to generate a baseline bot-likelihood probability. It utilizes Wei and Nguyen's bidirectional LSTM neural network, which incorporates GloVe word embeddings and achieves a 70% accuracy rate. User tweets are combined, tokenized, and converted into 25-dimensional GloVe Twitter embeddings. The model comprises three stacked bidirectional LSTM layers with an attention mechanism, ReLu activation, and a softmax activation layer that outputs the bot or human probability. Twibot-20 data is used for training, while predictions are generated for the Twibot-22 dataset based on the softmax layer output.

## INSTALLATION

- Prerequisite: You have anaconda/miniconda installed and set up on your machine. If not, please follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) depending on your system configuration.

- Create the environment from the given environment.yml file and activate it
    
    ```bash
    conda env create -f environment.yml
    conda activate tbd
    ```

- Place your twitter bearer token string in ```streamlit/token.txt```. This is used to get twitter handles for embedding the profile. Please contact us if you need a token.

- Start streamlit. A successful installation will open a browser window with the UI.
        
    ```bash
    streamlit run streamlit/main.py
    ```

## EXECUTION
- Produce embeddings of local network structure using [GraphWaveMachine](https://github.com/benedekrozemberczki/GraphWaveMachine)

    ```bash
    python3 src/main.py --input data/graphwave-database-twibot22-all-labels-text.txt --output output/twi22approx16sample64.csv --sample-number 64 --mechanism approximate --approximation 16    
    ```

- Store embeddings and NLP bot likelihoods

    ```bash
    python3 embeddings-and-probabilities.py --path ../twi22approx16sample64.csv
    ```
    
- Produce clusters and select clusters

    ```bash
    python3 embeddings-and-probabilities.py --path ../twi22approx16sample64.csv
    ```

- As the first step, we need to first load the cluster information into the system, using the given script. You can find the pkl file `clusters.pkl` inside clustering folder. The pkl file contains the cluster information for accounts along with their embeddings. The --reset flag is used to reset the DB. If you want to load new clusters into an existing DB, then set reset to 0. Following is a sample command to load the clusters into the system.

    ```bash
    cd scripts
    python load_clusters.py --file ../clustering/clusters.pkl --reset 1
    ```

    - After a successful run, you should see a message like this in the terminal

        ```bash
        Successfully loaded clusters into the DB
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

- Running this command should now open the UI. Change to dark mode and zoom in/out for a better user experience. 

    ```bash
    streamlit run streamlit/main.py
    ```

    - The UI should look something like this:

    ![UI](/docs/tbd_ui.png)

- The user interface consists of three primary elements:

    - Cluster Selection: This dropdown menu is pre-filled with clusters that have yet to be labeled. By choosing a cluster, you can view an interactive t-SNE reduced plot of that cluster, with the ability to zoom in and out to examine individual points. Hovering over these points will reveal the user ID.

    - Account Investigation: Once a cluster is selected, the right-hand section displays the user IDs of those within the chosen cluster. By selecting a user ID, their profile will be retrieved from the Twitter API and displayed in the interface. After examining the profile, you can label the user as either a bot or human and click "MARK" to add an entry to the review table at the bottom. This process can be repeated for multiple users, allowing for back-and-forth comparisons and labeling as necessary.

    - Saving to Database: After confidently labeling several users in a cluster as either bots or humans, click "SAVE ANNOTATIONS" to store the labels in the database. This action will update the cluster labels and apply the majority vote to any remaining accounts within the cluster. The interface will then automatically refresh, presenting the next unlabeled cluster for review.
    
    - This is a demo of a sample annotation session. 

    ![Demo](/docs/tbd_demo.gif)


