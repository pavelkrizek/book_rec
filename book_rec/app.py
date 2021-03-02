import pandas as pd
import streamlit as st
import torch

from book_rec import DATA_PATH
from book_rec.utils import get_annoy_similarity
from book_rec.utils import get_the_most_similar

pd.set_option("display.max_colwidth", None)

st.title("book recommendations")


@st.cache
def load_complete_lookup(path=DATA_PATH / "processed/complete_lookup.parquet"):
    return pd.read_parquet(path)


top_n = st.sidebar.number_input(
    "Select number of top book reccomandation",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
)
similarity_type = st.sidebar.radio(
    "Select similarity type", options=["angular", "euclidean", "manhattan", "hamming", "dot"], index=1
)


df = load_complete_lookup()
books = df["book_title"].tolist()
book_name = st.sidebar.selectbox("Select book", books)
embedings = torch.load(DATA_PATH / "processed/embedings.pt")
annoy_embedings = get_annoy_similarity(embedings, distance=similarity_type)
df_books = get_the_most_similar(df, annoy_embedings, book_name, top_n)
st.table(df_books.drop(columns="image_url_m"))
st.image(df_books["image_url_m"].tolist())  # caption=good_books)
