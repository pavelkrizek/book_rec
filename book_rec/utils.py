import json
from collections import namedtuple

import numpy as np
import pandas as pd
import requests
import torch
import tqdm
from annoy import AnnoyIndex

from book_rec import DATA_PATH
from book_rec import DESC


def create_description_embeddings_index(df, path=DATA_PATH, desc=DESC):
    import spacy

    df = df.copy()
    df_desc = pd.read_parquet(path / desc)
    df = df.merge(df_desc).drop_duplicates(subset=["book_auth"]).dropna(subset=["description", "categories"])

    isbn_data = namedtuple("w2c", ["isbn", "book_auth", "vector"])
    isbn_vector_list = []
    nlp = spacy.load("en_core_web_sm")
    for _, row in df.iterrows():
        vector = nlp(row.description).vector
        isbn_vector = isbn_data(isbn=row.isbn, book_auth=row.book_auth, vector=vector)
        isbn_vector_list.append(isbn_vector)

    embedings = np.vstack([isbn_vector.vector for isbn_vector in isbn_vector_list])
    book_index = {isbn_vector.book_auth: i for i, isbn_vector in enumerate(isbn_vector_list)}
    torch.save(torch.tensor(embedings), path / "processed/embedding_description.pt")
    with open(path / "processed/book_index_description.json", "w") as json_file:
        json.dump(book_index, json_file)

    return embedings, book_index


def load_book_index(path=DATA_PATH / "processed/book_index.json"):
    with open(path, "r") as json_file:
        return json.load(json_file)


def get_annoy_similarity(embeddings, distance="euclidean"):
    annoy = AnnoyIndex(embeddings.shape[1], distance)
    for i in range(embeddings.shape[0]):
        annoy.add_item(i, embeddings[i, :])
    annoy.build(n_trees=1)
    return annoy


def get_the_most_similar(df, annoy_embedings, book_name, top_n):
    index = df.loc[lambda df: df["book_title"] == book_name, "book_index"].max()
    top_n_index_book = annoy_embedings.get_nns_by_item(index, top_n + 1)[1:]
    return df.loc[
        lambda df: df["book_index"].isin(top_n_index_book),
        ["book_title", "book_author", "categories", "average_rating", "image_url_m"],
    ].reset_index(drop=True)


def get_additional_info(isbn_list):
    storage = []
    errors = 0
    isbn_data = namedtuple("ISBN", ["isbn", "average_rating", "ratings_count", "categories", "description"])
    for isbn in tqdm(isbn_list):
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        try:
            data = requests.get(url).json().get("items", None)[0]["volumeInfo"]
            storage.append(
                isbn_data(
                    isbn=isbn,
                    average_rating=data.get("averageRating", None),
                    ratings_count=data.get("ratingsCount", None),
                    categories=data.get("categories", None),
                    description=data.get("description", None),
                )
            )
        except Exception:
            errors += 1

    return storage
