import json
import logging

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from book_rec import DATA_PATH
from book_rec.dataset import BookDataset
from book_rec.engine import evaluate
from book_rec.engine import train
from book_rec.model import BookModel
from book_rec.preprocessing import save_lookup

CROSS_VALIDATE = False
N_EMBED = 50
TRAIN_BS = 32
VALID_BS = 8
EPOCHS = 9
FEATURES = ["user_id", "book_auth"]
TARGET = "book_rating"


def run(df, fold=None, save_embeddings=False, path=DATA_PATH):
    """
    Modeling pipeline function
    """
    df = df.copy()
    features = ["user_id", "book_auth"]
    feat_n = {}
    for col in features:
        lbl = LabelEncoder()
        df[col] = lbl.fit_transform(df[col])
        feat_n[col] = lbl

    if fold is not None:
        df_train = df[lambda x: x["kfold"] != fold].reset_index(drop=True)
        df_valid = df[lambda x: x["kfold"] == fold].reset_index(drop=True)
        valid_dataset = BookDataset(features=df_valid[features].values, targets=df_valid[TARGET].values)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, VALID_BS)
    else:
        df_train = df
    train_dataset = BookDataset(features=df_train[features].values, targets=df_train[TARGET].values)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, TRAIN_BS)

    n_users = len(feat_n["user_id"].classes_)
    n_books = len(feat_n["book_auth"].classes_)
    model = BookModel(n_users=n_users, n_books=n_books, n_embed=N_EMBED)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)

    logging.info("Training Model")
    best_mse = 100
    early_stopping_counter = 0
    for epoch in tqdm(range(EPOCHS)):
        train(train_data_loader, model, optimizer)
        if fold is not None:
            outputs, targets = evaluate(valid_data_loader, model)
            mse = mean_squared_error(targets, outputs)
            mae = mean_absolute_error(targets, outputs)
            logging.info(
                f"FOLD:{fold}, Epoch: {epoch}, Mean absolute error = {mae}, Mean squared error {mse}"
            )
            if best_mse > mse:
                best_mse = mse
            else:
                early_stopping_counter += 1
            if early_stopping_counter > 2:
                break

    if save_embeddings:
        torch.save(model.book_embed.weight, path / "processed/embedings.pt")
        book_index = {c: i for i, c in enumerate(feat_n["book_auth"].classes_)}
        with open(DATA_PATH / "processed/book_index.json", "w") as json_file:
            json.dump(book_index, json_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = pd.read_parquet(DATA_PATH / "processed/complete_data.parquet")
    if CROSS_VALIDATE:
        for fold in range(5):
            run(df, fold=fold)
    else:
        run(df, save_embeddings=True)
        save_lookup(df)
