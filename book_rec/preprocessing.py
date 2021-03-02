import logging
import zipfile

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from book_rec import DATA_PATH
from book_rec import DESC
from book_rec import FILE
from book_rec.utils import load_book_index


def read_data(data_path=DATA_PATH, filename=FILE):
    zf = zipfile.ZipFile(data_path / filename)
    csv_params = {"sep": ";", "error_bad_lines": False, "encoding": "latin-1"}
    dfs = {f.replace(".csv", ""): pd.read_csv(zf.open(f), **csv_params) for f in zf.namelist()}
    return dfs


def get_clean_data(dfs, min_books_user=5, min_ratings_book=20):
    df_raw = dfs["BX-Books"].merge(dfs["BX-Book-Ratings"]).merge(dfs["BX-Users"])

    df = (
        df_raw.copy()
        .rename(columns=lambda x: x.lower().replace("-", "_"))
        .assign(
            # drops 4 rows in wrong columns
            year_of_publication=lambda x: pd.to_numeric(x["year_of_publication"], errors="coerce"),
            book_title=lambda x: x["book_title"],
            book_author=lambda x: x["book_author"],
            book_auth=lambda x: x["book_title"] + "----" + x["book_author"],
        )
        .loc[lambda x: x["book_rating"] != 0]
        .groupby(["user_id"])
        .filter(lambda x: len(x) > min_books_user)
        .groupby(["book_auth"])
        .filter(lambda x: len(x) > min_ratings_book)
    )

    return df


def add_description_genre(df, path=DATA_PATH, desc=DESC):
    df_desc = pd.read_parquet(path / desc)
    return (
        df.merge(df_desc, how="left")
        .dropna(subset=["description", "categories"])
        .assign(categories=lambda df: df["categories"].apply(lambda x: x[0]))
    )


def add_embed_index(df, path=DATA_PATH / "processed/book_index.json"):
    return df.drop_duplicates(subset=["book_auth"]).assign(
        book_index=lambda df: df["book_auth"].map(load_book_index(path))
    )


def add_kfolds_col(df):
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["book_rating"].values
    kf = StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
    return df


def save_lookup(df, path=DATA_PATH / "processed"):
    df.pipe(add_embed_index).to_parquet(path / "complete_lookup.parquet")


if __name__ == "__main__":
    logging.info("Reading data")
    dfs = read_data()
    df = get_clean_data(dfs).pipe(add_description_genre).pipe(add_kfolds_col)
    save_lookup(df)
    logging.info("Storing complete preprocessed data")
    df.to_parquet(DATA_PATH / "complete_data.parquet")
