from sentence_transformers import SentenceTransformer, util
import pandas as pd
import streamlit as st
import os
import shutil
import numpy as np
import time
import keyboard
import psutil
import fitz
import re
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


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


def extract_abstract(article, char_max=1000):
    """Extract the abstract from a pdf file

    Parameters
    ----------
    path : str
        Path to the pdf file
    char_max : int, optional
        Maximum number of characters to extract if introduction not found, by default 1000

    Returns
    -------
    str
        Abstract of the pdf file
    """
    with fitz.open(stream=article.read(), filetype="pdf") as doc:
        text = ""
        # Add two extra pages to the text
        for i in range(3):
            text += doc[i].get_textpage().extractText().lower()
        # Find the abstract with white spaces inbetween characters
        ret = None
        s = 0
        while not ret and s < 3:
            search = "a.{"+str(s)+"}b.{"+str(s)+"}s.{"+str(s)+"}t.{" + \
                str(s)+"}r.{"+str(s)+"}a.{"+str(s)+"}c.{"+str(s)+"}t"
            ret = re.search(search, text)
            s += not ret
        abstract_unbounded = re.split(search, text)[1]
        # Find the introduction with white spaces inbetween characters
        ret = None
        s = 0
        while not ret and s < 3:
            search = "i.{"+str(s)+"}n.{"+str(s)+"}t.{"+str(s)+"}r.{"+str(s)+"}o.{"+str(s)+"}d.{"+str(s)+"}u.{"+str(s) +\
                "}c.{"+str(s)+"}t.{"+str(s)+"}i.{"+str(s)+"}o.{"+str(s)+"}n"
            ret = re.search(search, text)
            i += not ret
        if not ret:
            abstract = abstract_unbounded[:char_max]
        abstract = re.split(search, abstract_unbounded)[0]
        return abstract


embeddings = pd.read_pickle("embeddings.pkl")
index = pd.read_excel("kb_articles/AAA_index.xlsx")
chosen_embedding = embeddings["msmarco-distilbert-base-tas-b"]
# write the chosen embedding to a dataframe
embedding_space = pd.DataFrame(
    chosen_embedding.tolist(), index=chosen_embedding.index)
embedding_space.columns = [f"dim_{i}" for i in embedding_space.columns]
# dimensionality reduction to 2D
preprocessing_pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
preprocessing_pipe.fit(embedding_space)
embedding_space_2D = pd.DataFrame(preprocessing_pipe.transform(embedding_space)).rename(
    columns={0: "PC1", 1: "PC2"})
clusterer = KMeans(n_clusters=3, random_state=0)
clusterer.fit(embedding_space_2D)

model = SentenceTransformer("msmarco-distilbert-base-tas-b")


def show_close_articles(abstract, n):
    # Get the centroids of the clusters
    centroids = clusterer.cluster_centers_
    # Embed the abstract
    emb_abstract = model.encode(abstract, convert_to_tensor=False)
    red_emb_abstract = preprocessing_pipe.transform([emb_abstract])
    # Get the closest articles
    knn = NearestNeighbors(n_neighbors=n, metric="euclidean")
    knn.fit(embedding_space_2D)
    distances, indices = knn.kneighbors(red_emb_abstract)
    # Plot the figure
    fig, ax = plt.subplots()
    ax.scatter(
        embedding_space_2D.PC1, embedding_space_2D.PC2, c=clusterer.labels_,
        cmap="viridis", alpha=0.5, s=15)
    ax.scatter(
        centroids[:, 0], centroids[:, 1], c="black", s=100, marker="X")
    ax.scatter(
        red_emb_abstract[:, 0], red_emb_abstract[:, 1], c="red", s=300, marker="*")
    drawn_circle = plt.Circle(
        (red_emb_abstract[:, 0], red_emb_abstract[:, 1]), distances.max(), color='r', fill=False)
    fig.gca().add_artist(drawn_circle)
    ax.set_xlabel('First dimension of the embedding space')
    ax.set_ylabel('Second dimension')
    ax.set_title('Clusters of the knowledge base')
    return fig, index.loc[list(indices[0]), ["title", "abstract", "path"]]


if __name__ == '__main__':

    st.set_page_config(
        page_title="Reacfin Knowledge Base",
        page_icon=":books:")

    st.write("""
    # Reacfin Knowledge Base
    
    Welcome to the Reacfin Knowledge Base! Thanks to our new semantic search engine,
    you can now search through our collection of articles and retrieve the most relevant
    ones for your query.

    *developed by: [Mathieu Demarets](https://www.linkedin.com/in/mathieudemarets/)*

    ---
    """)
    if "article" not in st.session_state:
        st.session_state["article"] = None

    query = st.text_input(
        "What do you want to know?",
        "What are P2P loans?")
    n = st.slider("Number of articles to retrieve:", 1, 10, 1)
    softmax = st.checkbox("Use softmax for ranking?")
    output_path = st.text_input(
        "Where do you want to store the pdf?",
        "C:/Users/User/Desktop/Requested Articles")
    if st.button("Search"):
        retrieved = SemanticSearchEngine_N(
            query, n=n, softmax=softmax, output_path=output_path)
        st.write(retrieved)

    def change_article():
        st.session_state["article"] = "uploaded"

    st.write("""

    ---

    ## Find similar articles

    Upload your PDF article to find similar ones in our collection. The search engine will use the abstract
    to perform the search.
    """)
    uploaded_pdf = st.file_uploader(
        "Upload your PDF file", type="pdf", on_change=change_article)
    if st.session_state["article"] == "uploaded":
        st.success("File uploaded successfully!")
        abstract = extract_abstract(uploaded_pdf)
        n_close = st.slider("Number of similar articles in radius:", 1, 5, 2)
        if st.button("Show similar articles"):
            fig, close_articles = show_close_articles(abstract, n_close)
            st.pyplot(fig)
            st.write(close_articles)

    # Solution to shut down the app inspired by: https://discuss.streamlit.io/t/close-streamlit-app-with-button-click/35132/4
    if st.sidebar.button("Shut Down"):
        # Give a bit of delay for user experience
        st.sidebar.write(
            "Thank you for using our search engine!\nShutting down...")
        time.sleep(3)
        # Close streamlit browser tab
        keyboard.press_and_release('ctrl+w')
        # Terminate streamlit python process
        pid = os.getpid()
        p = psutil.Process(pid)
        p.terminate()
