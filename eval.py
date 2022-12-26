import argparse
import datetime
from glob import glob
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from train import *
from trainer import *
from metrics import *

if __name__ == "__main__":
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

    args.stock_list = args.stock_list[:10]
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


    mix_pv_val = gen_dataset(
        dataframe_dict={"price": prices, "tweet": tweets},
        start_date="2016-01-26",
        end_date="2016-06-31",
    )

    valset = MixDataset(
        mode="val",
        data=mix_pv_val,
        window_num=args.window,
        max_tweet_num=args.max_tweet_num,
        max_tweet_len=args.max_tweet_len,
        stock_list=args.stock_list,
    )
   
    valsampler = RandomSampler(valset)
    
    valloader = DataLoader(
        valset, sampler=valsampler, batch_size=args.batch_size, drop_last=True
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


    val_preds, val_labelss, val_metrics, loss_val, val_attn = get_predictions(
            model, valloader, edge_index, edge_type, validation=True, eval_mode=True
        )

    print(
        "F1: %.3f \n" % (np.mean(val_metrics["f1"])),
        "Recall: %.3f \n" % (np.mean(val_metrics["recall"])),
        "Precision: %.3f \n" % (np.mean(val_metrics["precision"])),
        "Accuracy: %.3f \n " % (np.mean(val_metrics["accuracy"])),
        "MCC: %.3f \n " % (np.mean(val_metrics["mcc_score"])),
        )