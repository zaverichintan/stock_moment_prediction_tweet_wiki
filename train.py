from __future__ import division
from __future__ import print_function

import argparse
import datetime
from glob import glob
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, RandomSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import pickle
import random
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from trainer import *
from tqdm import tqdm
from dataloader import MixDataset
from model import MANSF
from utils import build_wiki_relation, accuracy

def gen_dataset(dataframe_dict: dict, start_date: str, end_date: str):
    """generate data with different date range"""
    for name in dataframe_dict.keys():
        dataframe_dict[name] = (
            dataframe_dict[name].drop_duplicates().reset_index(drop=True)
        )
        dataframe_dict[name] = dataframe_dict[name][
            (dataframe_dict[name]["date"].notnull())
            & (dataframe_dict[name]["date"] >= start_date)
            & (dataframe_dict[name]["date"] <= end_date)
        ].reset_index(drop=True)
    prices, tweets = dataframe_dict["price"], dataframe_dict["tweet"]
    prices = prices.sort_values("date").reset_index(drop=True)
    tweets = (
        tweets.groupby(["stock", "date"], as_index=False)
        .agg({"text": "\n".join})
        .fillna("")
        .reset_index(drop=True)
    )
    mix = (
        pd.merge(prices, tweets, on=["stock", "date"], how="left")
        .fillna("")
        .reset_index(drop=True)
    )
    mix_pv = pd.pivot(mix, index="date", columns="stock").reset_index()
    mix_pv["text"] = mix_pv["text"].fillna("")
    for col in ["movement_perc", "high", "low", "close"]:
        mix_pv[col] = mix_pv[col].fillna(0.0)
    return mix_pv



if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument(
        "--fastmode",
        action="store_true",
        default=False,
        help="Validate during training pass.",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        default=False,
        help="GAT with sparse version or not.",
    )
    parser.add_argument("--seed", type=int, default=14, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of batch size to train."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units.")
    parser.add_argument(
        "--nb_heads", type=int, default=8, help="Number of head attentions."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.38, help="Dropout rate (1 - keep probability)."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    parser.add_argument("--patience", type=int, default=100, help="Patience")
    parser.add_argument("--window", type=int, default=5, help="Window of trading day")
    parser.add_argument(
        "--max_tweet_num",
        type=int,
        default=5,
        help="Max number of tweets used for training per day per stock",
    )
    parser.add_argument(
        "--max_tweet_len", type=int, default=30, help="Max length of tweets embedding"
    )
    parser.add_argument(
        "--text_ft_dim", type=int, default=384, help="Dimension of tweets embedding"
    )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
        args.mps_device = None
    else:
        args.mps_device = torch.device("mps")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    stock_dir = "stocknet-dataset/price/preprocessed"
    tweet_dir = "stocknet-dataset/tweet_preprocessed"
    rels_dir = "Temporal_Relational_Stock_Ranking/data/relation/wikidata"
    market_names = ["NASDAQ", "NYSE"]

    args.stock_list = sorted(
        list(
            set(
                [path.split("/")[-1].replace(".txt", "") for path in glob(stock_dir + "/*")]
            )
            & set([path.split("/")[-1].split('.')[0] for path in glob(tweet_dir + "/*")])


        )
    )
    # args.stock_list = ['AAPL', 'BA', 'BAC']
    args.stock_list = args.stock_list[:20]
    args.n_stock = len(args.stock_list)  # the number of stocks
    n_day = 5  # the backward-looking window T
    n_tweet = 5  # max num of tweets per day, I suppose 1 tweet per stock per day
    n_price_feat = 3  # price feature dim  (normalized high/low/close)
    n_tweet_feat = 384  # text embedding dim

    prices, tweets = pd.DataFrame(), pd.DataFrame()
    for stock in tqdm(args.stock_list):
        _p = pd.read_table(os.path.join(stock_dir, f"{stock}.txt"), header=None)
        _t = pd.read_json(os.path.join(tweet_dir, f"{stock}.json"), orient="records", lines=True)
        
        _t['text'] = [' '.join(map(str, l)) for l in _t['text']]
        
        _p["stock"], _t["stock"] = stock, stock
        prices, tweets = pd.concat([prices, _p]), pd.concat([tweets, _t])


    prices.columns = [
        "date",
        "movement_perc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "stock",
    ]
    prices = prices.drop(["open", "volume"], axis=1)
    tweets["date"] = tweets.created_at.apply(lambda x: x.date())
    tweets.date = tweets.date.astype(str)

    mix_pv_train = gen_dataset(
        dataframe_dict={"price": prices, "tweet": tweets},
        start_date="2014-01-01",
        end_date="2015-06-30",
    )
    # mix_pv_train = gen_dataset(
    #     dataframe_dict={"price": prices, "tweet": tweets},
    #     start_date="2015-06-20",
    #     end_date="2015-06-30",
    # )
    mix_pv_val = gen_dataset(
        dataframe_dict={"price": prices, "tweet": tweets},
        start_date="2015-07-26",
        end_date="2015-12-31",
        # end_date="2015-07-31",
    )
    mix_pv_test = gen_dataset(
        dataframe_dict={"price": prices, "tweet": tweets},
        start_date="2016-01-26",
        end_date="2016-03-31",
        # end_date="2016-01-31",
    )

    # preprocess relation data & load
    adj = build_wiki_relation(rels_dir, market_names, args.stock_list)
    edge_index = torch.index_select(torch.nonzero(adj).t(), 0, torch.tensor([0, 1]))
    edge_type = torch.index_select(torch.nonzero(adj).t(), 0, torch.tensor([2])).squeeze(0)

    model = MANSF(
        nfeat=64,
        nhid=args.hidden,
        nrel=adj.shape[2],
        nclass=2,
        dropout=args.dropout,
        nheads=args.nb_heads,
        alpha=args.alpha,
        stock_num=args.n_stock,
        text_ft_dim=args.text_ft_dim,
    )
    # model = nn.DataParallel(model, device_ids=[0, 1])
    if args.cuda:
        model.cuda()
        edge_index = edge_index.type(torch.LongTensor).cuda()
        edge_type = edge_type.type(torch.LongTensor).cuda()
        # adj = adj.type(torch.LongTensor).cuda()
        args.device = "cuda"
    
    # if args.mps_device:
    #     mps_device = torch.device("mps")
    #     model.to(mps_device)
    #     edge_index = edge_index.type(torch.LongTensor).to(mps_device)
    #     edge_type = edge_type.type(torch.LongTensor).to(mps_device)
    #     # adj = adj.type(torch.LongTensor).cuda()
    #     args.device = mps_device 

    trainset = MixDataset(
        mode="train",
        data=mix_pv_train,
        window_num=args.window,
        max_tweet_num=args.max_tweet_num,
        max_tweet_len=args.max_tweet_len,
        stock_list=args.stock_list,
    )
    valset = MixDataset(
        mode="val",
        data=mix_pv_val,
        window_num=args.window,
        max_tweet_num=args.max_tweet_num,
        max_tweet_len=args.max_tweet_len,
        stock_list=args.stock_list,
    )
    testset = MixDataset(
        mode="test",
        data=mix_pv_test,
        window_num=args.window,
        max_tweet_num=args.max_tweet_num,
        max_tweet_len=args.max_tweet_len,
        stock_list=args.stock_list,
    )

    trainsampler = RandomSampler(trainset)
    valsampler = RandomSampler(valset)
    testsampler = RandomSampler(testset)
    trainloader = DataLoader(
        trainset, sampler=trainsampler, batch_size=args.batch_size, drop_last=True
    )
    valloader = DataLoader(
        valset, sampler=valsampler, batch_size=args.batch_size, drop_last=True
    )
    testloader = DataLoader(
        testset, sampler=testsampler, batch_size=args.batch_size, drop_last=True
    )
    print(len(trainset), len(valset), len(testset))

    train(args, model, trainloader, valloader, edge_index, edge_type)
    print("Optimization Finished!")
    results = test_dict()
    print(results)
