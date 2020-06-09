"""
Crated on June 4, 2020
Author: Joe Tawea

To Do:
    try-except to check files exist when imorting them
    try-except to check folder before saving model and df
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.layers import Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_review(review_pkl):

    review_df = pd.read_pickle(review_pkl)

    # (Optional) Scaled down data
    ct_df = review_df.groupby(["reviewerID"]).count().sort_values(by=["overall"])
    x_users = ct_df[(ct_df.overall >= 10) & (ct_df.overall <= 28)]["overall"]
    scaled_df = review_df[review_df["reviewerID"].isin(x_users.index)]
    # Reset review_df to make this code section optional
    review_df = scaled_df

    return review_df


def load_product(product_pkl):

    product_df = pd.read_pickle(product_pkl)

    return product_df


def build_model(X):

    # Hyper-parameter - Embedding dimension
    D = 10

    U = len(set(X[:, 0]))  # number of users
    P = len(set(X[:, 1]))  # number of products

    u = Input(shape=(1,))
    p = Input(shape=(1,))
    u_emb = Embedding(U, D)(u)  # (num_samples, 1, D)
    p_emb = Embedding(P, D)(p)  # (num_samples, 1, D)
    u_emb = Flatten()(u_emb)  # (num_samples, D)
    p_emb = Flatten()(p_emb)  # (num_samples, D)
    x = Concatenate()([u_emb, p_emb])  # (num_samples, 2D)
    x = Dense(512, activation="relu")(x)
    x = Dense(1)(x)
    model = Model(inputs=(u, p), outputs=x)

    return model


def export_model(model, model_name):

    print("... save model")
    model.save(model_name)
    print(f"Done save model {model_name}\n")


def save_input_df(df, df_name):

    print("... save input df")
    df.to_pickle(df_name, protocol=3)
    print(f"Done save input df {df_name}\n")


if __name__ == "__main__":

    review_pkl = "./data/Book_review_df.pkl"
    product_pkl = "./data/meta_Book_product_df.pkl"

    saved_model = "./saved_model_input/tf_model.h5"
    saved_input = "./saved_model_input/review_cat_df.pkl"

    print("Load review and product data...")
    review_df = load_review(review_pkl)
    product_df = load_product(product_pkl)
    print("Done load review and product data\n")

    print("Consolidate the data")
    review_cat_df = pd.merge(review_df, product_df, how="inner", on=["asin"])
    review_cat_df["userID"] = review_cat_df["reviewerID"].astype("category").cat.codes
    review_cat_df["productID"] = review_cat_df["asin"].astype("category").cat.codes

    print("Train-Test split\n")
    X = np.concatenate(
        (
            review_cat_df["userID"].values.reshape(-1, 1),
            review_cat_df["productID"].values.reshape(-1, 1),
        ),
        axis=1,
    )
    y = review_cat_df["overall"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Build the model")
    model = build_model(X)
    model.compile(loss="mse", optimizer=SGD(lr=0.02, momentum=0.9))

    print("...Traing start time = ", datetime.now().strftime("%H:%M:%S"))
    r = model.fit(
        x=[X_train[:, 0], X_train[:, 1]],
        y=y_train,
        epochs=12,
        batch_size=1024,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    )
    print("   Training end time = ", datetime.now().strftime("%H:%M:%S"), "\n")

    export_model(model, saved_model)
    save_input_df(review_cat_df, saved_input)
    print("Done save model and input data df")
