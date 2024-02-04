from sentence_transformers import SentenceTransformer, util
import pandas as pd
import streamlit as st
import os
import shutil
import numpy as np
import time
import keyboard
import psutil


def SemanticSearchEngine_N(
        query, n=3, softmax=False, output_path=None,
        index="kb_articles/AAA_index.xlsx", embeddings="embeddings.pkl",
        verbose=True):
    """Search engine using the msmarco-distilbert-base-tas-b embedding and the dot product

    Parameters
    ----------
    query : str
        Query to search
    n : int, optional
        Number of articles to retrieve, by default 3
    softmax : bool, optional
        Use softmax instead of the score, by default False
    output_path : str, optional
        Path to save the article, by default None
    index : str, optional
        Path to the index file, by default "kb_articles/AAA_index.xlsx"
    embeddings : str, optional
        Path to the embeddings file, by default "embeddings.pkl"
    verbose : bool, optional
        Print the retrieved articles, by default True

    Returns
    -------
    pd.DataFrame
        Dataframe of the retrieved articles (title, score, pages, abstract, path)
    """
    embeddings = pd.read_pickle(embeddings)
    model = SentenceTransformer("msmarco-distilbert-base-tas-b")
    model.max_seq_length = 512
    embedded_query = model.encode(query, convert_to_tensor=False)
    embedded_corpus = embeddings["msmarco-distilbert-base-tas-b"]

    # Get the similarity scores
    sim = util.dot_score(embedded_query, embedded_corpus)
    output = pd.DataFrame(sim).T
    output["Key"] = output.index
    output.columns = ["Score", "Key"]
    rank = "Score"

    # Use softmax if needed
    if softmax:
        def get_softmax(scores):
            return np.exp(scores)/np.exp(scores).sum()
        output["Softmax"] = get_softmax(output.Score.to_list())
        rank = "Softmax"
    output = output.sort_values(by=rank, ascending=False).head(n)

    # Copy the articles to the output_path if needed
    if output_path is not None:
        index = pd.read_excel(index)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        retrieved = pd.merge(output, index, left_on="Key", right_index=True)
        i = 0
        for _, row in retrieved.iterrows():
            i += 1
            path = row.path
            path_out = os.path.join(output_path, f"{row.title}.pdf")
            path_out = shutil.copy(path, path_out)
            if verbose:
                print(
                    f"\nTop {i}: {output.iloc[i-1][rank]:.2f}\n'{row.title}' copied to:\n   -> '{path_out}'")
    return retrieved.loc[:, ["title", rank, "pages", "abstract", "path"]]


if __name__ == '__main__':
    st.write("""
    # Reacfin Knowledge Base
    
    Welcome to the Reacfin Knowledge Base! Thanks to our new semantic search engine,
    you can now search through our collection of articles and retrieve the most relevant
    ones for your query.

    *developed by: [Mathieu Demarets](https://www.linkedin.com/in/mathieudemarets/)*

    ---
    """)
    query = st.text_input(
        "What do you want to know?",
        "What are good performance metrics for credit default models?")
    n = st.slider("Number of articles to retrieve:", 1, 10, 1)
    softmax = st.checkbox("Use softmax for ranking?")
    output_path = st.text_input(
        "Where do you want to store the pdf?",
        "C:/Users/User/Desktop/Requested Articles")
    if st.button("Search"):
        retrieved = SemanticSearchEngine_N(
            query, n=n, softmax=softmax, output_path=output_path)
        st.write(retrieved)

    # Solution to shut down the app inspired by: https://discuss.streamlit.io/t/close-streamlit-app-with-button-click/35132/4
    if st.button("Shut Down"):
        # Give a bit of delay for user experience
        st.write("Thank you for using our search engine!\nShutting down...")
        time.sleep(3)
        # Close streamlit browser tab
        keyboard.press_and_release('ctrl+w')
        # Terminate streamlit python process
        pid = os.getpid()
        p = psutil.Process(pid)
        p.terminate()
