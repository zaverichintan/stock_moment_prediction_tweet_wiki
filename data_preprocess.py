import datetime
from glob import glob
import numpy as np
import os
import json
import pprint as pp #Pretty printer
import shutil

stock_dir = "stocknet-dataset/price/preprocessed"
tweet_dir = "stocknet-dataset/tweet/preprocessed"
out_tweet_processed = "stocknet-dataset/tweet_preprocessed"

stock_list = sorted(
    list(
        set(
            [path1.split("/")[-1].replace(".txt", "") for path1 in glob(stock_dir + "/*")]
        )
        & set([path2.split("/")[-1] for path2 in glob(tweet_dir + "/*")])
    )
)

for stock in stock_list:
    out_path = os.path.join(out_tweet_processed, f"{stock}.json")

    with open(out_path, 'wb') as outfile:
        for filename in glob(os.path.join(tweet_dir, f"{stock}", "*")):
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)