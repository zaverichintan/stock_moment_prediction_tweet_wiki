import numpy as np 
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
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mansf-stocknet")

from tqdm import tqdm
mps_device = torch.device("mps")

def train(args, model, trainloader, valloader, edge_index, edge_type):
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # loss_fn = nn.BCELoss().cuda()
    loss_fn = nn.BCELoss()
    # loss_fn = nn.BCELoss().to(mps_device)
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        loss_train = 0.0
        metrics = {k: [] for k in ["accuracy", "f1", "recall", "precision"]}
        # metrics = {k: [] for k in ["accuracy", "roc_auc", "f1", "recall", "precision"]}
        for i, data in enumerate(tqdm(trainloader, desc="Training")):
            text_tensor, price_tensor, label_tensor = data
            optimizer.zero_grad()

            # forward pass
            output = model(text_tensor, price_tensor, edge_index, edge_type)
            _output, _label = (
                output.argmax(axis=1).unsqueeze(1).float(),
                label_tensor.squeeze(0).float(),
            )
            
            _loss_train = Variable(loss_fn(_output, _label), requires_grad=True)
            metrics["accuracy"].append(accuracy_score(_output.cpu(), _label.cpu()))
            # metrics["roc_auc"].append(roc_auc_score(_output.cpu(), _label.cpu()))
            metrics["f1"].append(f1_score(_output.cpu(), _label.cpu()))
            metrics["recall"].append(recall_score(_output.cpu(), _label.cpu()))
            metrics["precision"].append(precision_score(_output.cpu(), _label.cpu()))

            # backward
            _loss_train.backward()
            optimizer.step()
            loss_train += _loss_train.mean().item()
            # torch.cuda.empty_cache()
        
        # epoch trainset
        print("[Epoch %s] Train Loss: %.3f \n" % (epoch, loss_train))
        # _, labelss, metrics = get_predictions(clf, trainloader, compute_acc=True, compute_loss=False)
        print(
            "    Train F1: %.3f \n" % (np.mean(metrics["f1"])),
            "    Train Recall: %.3f \n" % (np.mean(metrics["recall"])),
            "    Train Precision: %.3f \n" % (np.mean(metrics["precision"])),
            # "    Train ROC AUC: %.3f \n " % (np.mean(metrics["roc_auc"])),
            "    Train Accuracy: %.3f \n " % (np.mean(metrics["accuracy"])),
        )
        # valset
        val_preds, val_labelss, val_metrics, loss_val, val_attn = get_predictions(
            model, valloader, edge_index, edge_type, validation=True
        )
        print("[Epoch %s] Val Loss: %.3f \n" % (epoch, loss_val))
        print(
            "    Val F1: %.3f \n" % (np.mean(val_metrics["f1"])),
            "    Val Recall: %.3f \n" % (np.mean(val_metrics["recall"])),
            "    Val Precision: %.3f \n" % (np.mean(val_metrics["precision"])),
            # "    Val ROC AUC: %.3f \n " % (np.mean(val_metrics["roc_auc"])),
            "    Val Accuracy: %.3f \n " % (np.mean(val_metrics["accuracy"])),
        )
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": loss_train, "Validation": loss_val},
            epoch * len(trainloader) + i,
        )
        for name in metrics:
            writer.add_scalars(
                f"Training vs. Validation {name.upper()}",
                {
                    "Training": np.mean(metrics[name]),
                    "Validation": np.mean(val_metrics[name]),
                },
                epoch * len(trainloader) + i,
            )
        np.save("output/preds-mansf-stocknet", val_preds.cpu())
        np.save("output/attns-mansf-stocknet", val_attn.cpu())
        torch.save(model.state_dict(), "finetuned/mansf-stocknet.pth")
        writer.flush()

def get_predictions(model, dataloader, edge_index, edge_type, validation=False, eval_mode=False):
    predictions, attns, labelss = None, None, None
    losses = 0.0
    metrics = {k: [] for k in ["accuracy", "f1", "recall", "precision", "mcc_score"]}
    # metrics = {k: [] for k in ["accuracy", "roc_auc", "f1", "recall", "precision"]}
    # loss_fn = nn.BCELoss().cuda()
    loss_fn = nn.BCELoss()
    # loss_fn = nn.BCELoss().to(mps_device)
    with torch.no_grad():
        for data in tqdm(dataloader):

            if next(model.parameters()).is_cuda == False:
                if validation:
                    text_tensor, price_tensor, label_tensor = data
                else:
                    text_tensor, price_tensor = data
                output, _ = model(
                    text_tensor, price_tensor, edge_index, edge_type, return_attn=True
                )
                if validation:
                    _output, _label = (
                        output.argmax(axis=1).unsqueeze(1).float(),
                        label_tensor.squeeze(0).float(),
                    )
                    _loss = loss_fn(_output, _label)
                    losses += _loss.mean().item()
                    metrics["accuracy"].append(
                        accuracy_score(_output.cpu(), _label.cpu())
                    )
                    metrics["f1"].append(f1_score(_output.cpu(), _label.cpu()))
                    metrics["recall"].append(recall_score(_output.cpu(), _label.cpu()))
                    metrics["precision"].append(
                        precision_score(_output.cpu(), _label.cpu())
                    )
                    metrics["mcc_score"].append(
                        matthews_corrcoef(_output.cpu(), _label.cpu())
                    )
                torch.cuda.empty_cache()
                if predictions is None:
                    predictions, attns = output, _[1]
                    if validation:
                        labelss = label_tensor
                else:
                    predictions = torch.cat((predictions, output))
                    attns = torch.cat((attns, _[1]))
                    if validation:
                        labelss = torch.cat((labelss, label_tensor))
    if validation:
        return predictions, labelss, metrics, losses, attns

    return predictions, attns


# def test_dict():
#     pred_dict = dict()
#     with open("label_data.p", "rb") as fp:
#         true_label = pickle.load(fp)
#     with open("price_feature_data.p", "rb") as fp:
#         feature_data = pickle.load(fp)
#     with open("text_feature_data.p", "rb") as fp:
#         text_ft_data = pickle.load(fp)
#     model.eval()
#     test_acc = []
#     test_loss = []
#     li_pred = []
#     li_true = []
#     for dates in feature_data.keys():
#         # test_text = torch.tensor(text_ft_data[dates], dtype=torch.float32).cuda()
#         test_text = torch.tensor(text_ft_data[dates], dtype=torch.float32).to(mps_device)
#         # test_price = torch.tensor(feature_data[dates], dtype=torch.float32).cuda()
#         test_price = torch.tensor(feature_data[dates], dtype=torch.float32).to(mps_device)
#         # test_label = torch.LongTensor(true_label[dates]).cuda()
#         test_label = torch.LongTensor(true_label[dates])
#         output = model(test_text, test_price, edge_index, edge_type)
#         output = F.softmax(output, dim=1)
#         # pred_dict[dates] = output.cpu().detach().numpy()
#         pred_dict[dates] = output.detach().numpy()
#         loss_test = F.nll_loss(output, torch.max(test_label, 1)[0])
#         acc_test = accuracy(output, torch.max(test_label, 1)[1])
#         a = torch.max(output, 1)[1].cpu().numpy()
#         b = torch.max(test_label, 1)[1].cpu().numpy()
#         li_pred.append(a)
#         li_true.append(b)
#         test_loss.append(loss_test.item())
#         test_acc.append(acc_test.item())
#     iop = f1_score(
#         np.array(li_true).reshape((-1,)),
#         np.array(li_pred).reshape((-1,)),
#         average="micro",
#     )
#     mat = matthews_corrcoef(
#         np.array(li_true).reshape((-1,)), np.array(li_pred).reshape((-1,))
#     )
#     print(
#         "Test set results:",
#         "loss= {:.4f}".format(np.array(test_loss).mean()),
#         "accuracy= {:.4f}".format(np.array(test_acc).mean()),
#         "F1 score={:.4f}".format(iop),
#         "MCC = {:.4f}".format(mat),
#     )
#     with open("pred_dict.p", "wb") as fp:
#         pickle.dump(pred_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
#     return iop, mat
