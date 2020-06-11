"""
Crated on June 8, 2020
Author: Joe Tawea

To Do:
    try-except to check folder before loading
"""

import os
import numpy as np
import pandas as pd
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

top_n = 10
target_user = "A324TTUBKTN73A"

model_name = "./saved_model_input/tf_model.h5"
input_df = "./saved_model_input/review_cat_df.pkl"

print("Load model and input data...")
model = keras.models.load_model(model_name)
review_cat_df = pd.read_pickle(input_df)
print("Done load model and input data\n")

print("Get product history and new products for the user")
user_id = review_cat_df[review_cat_df["reviewerID"] == target_user]["userID"].values[0]
bought = review_cat_df[review_cat_df["reviewerID"] == target_user]["productID"]
not_bought = np.unique(
    review_cat_df[~review_cat_df["productID"].isin(bought)]["productID"].values
)

print("Predict the product ratings")
predicted_rate = model.predict(x=[np.full((not_bought.shape[0],), user_id), not_bought])

print("Aggregate data for product recommendation\n")
prod_rating_df = pd.DataFrame({"predicted_rate": predicted_rate.ravel()})
prod_rating_df["productID"] = not_bought

# Drop duplicates due to other users
prod_title_df = review_cat_df.drop_duplicates(
    subset=["productID", "title"], keep="first"
)
title_rating_df = pd.merge(prod_rating_df, prod_title_df, how="inner", on=["productID"])

print(f"List of {top_n} Recommendation")
top_n_df = title_rating_df.sort_values(by=["predicted_rate"], ascending=False)[
    ["title", "asin", "category"]
].head(top_n)
print("ASIN\t\t", "Title")
for index, row in top_n_df.iterrows():
    print(row["asin"], "\t", row["title"])
