# -*- coding: utf-8 -*-
"""
April 2023
Updated code by Madelyn Scandlen
Original code by Shangbin Feng et al. found at https://github.com/LuoUndergradXJTU/TwiBot-22/blob/master/src/Wei/Twibot-20/data_processor.py
Inspired by Feng Wei & Uyen Trang Nguyen
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from data_processor import DataProcessor
torch.manual_seed(100)

vocab_size = 50000
embedding_size = 25
num_classes = 2
sentence_max_len = 64
hidden_size = 16

num_layers =3
num_directions = 2
lr = 1e-3
batch_size = 16
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMModel(nn.Module):
    def __init__(self, embedding_size,hidden_size, num_layers, num_directions, num_classes):
        super(BiLSTMModel, self).__init__()

        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions


        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = num_layers, bidirectional = (num_directions == 2))
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(hidden_size, num_classes)
        self.act_func = nn.Softmax(dim=1)

    def forward(self, x):
        #x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)         #[sentence_length, batch_size, embedding_size]

        batch_size = x.size(1)

        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

        #h_n, c_n [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]

        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]

        attention_w = self.attention_weights_layer(h_n)  #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]

        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  #[batch, 1, seq_len]

        x = torch.bmm(softmax_w, out)  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch, hidden_size]
        x = self.linear(x)
        x = self.act_func(x)
        return x

def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    nump = 0.0
    all_preds = []
    all_ids = []

    for datas, labels, ids in tqdm(test_loader):
        ids = ids.to(device)
        datas = datas.to(device)
        labels = labels.to(device)

        preds = model(datas)
        loss = loss_func(preds, labels)


        labels = torch.argmax(labels, dim=1)
        for j in range(len(preds)):
            m = labels[j]
            all_preds.append(preds[j][m].item())

        for i in ids:
            all_ids.append(i.item())

        loss_val += loss.item() * datas.size(0)

        preds = torch.argmax(preds, dim=1)
        corrects += torch.sum(preds == labels).item()
        TP += torch.sum((preds == 1) & (labels == 1)).item()
        TN += torch.sum((preds == 0) & (labels == 0)).item()
        FP += torch.sum((preds == 1) & (labels == 0)).item()
        FN += torch.sum((preds == 0) & (labels == 1)).item()
        nump += torch.sum((preds == 1)).item()

    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    test_acc2 = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    F1 = 2 * precision*recall / (precision + recall + 1e-6)
    nump = nump / len(test_loader.dataset)

    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    print("Test Acc2: {}".format(test_acc2))
    print("Test precision: {}, Test recall: {}".format(precision, recall))
    print("Test  F1: {}, positive rate: {}".format(F1, nump))
    return test_acc, all_preds, all_ids

def train(model, train_loader, test_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in tqdm(range(epochs)):
        print("train epoch ", epoch)
        model.train()
        loss_val = 0.0
        corrects = 0.0
        TP = 0.0
        TN = 0.0
        FP = 0.0
        FN = 0.0
        nump = 0.0
        train_acc = 0.0
        train_acc2 = 0.0

        for datas, labels, ids in tqdm(train_loader):
            datas = datas.to(device)
            labels = labels.to(device)

            preds = model(datas)
            loss = loss_func(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item() * datas.size(0)

            #获取预测的最大概率出现的位置
            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            corrects += torch.sum(preds == labels).item()
            TP += torch.sum((preds == 1) & (labels == 1)).item()
            TN += torch.sum((preds == 0) & (labels == 0)).item()
            FP += torch.sum((preds == 1) & (labels == 0)).item()
            FN += torch.sum((preds == 0) & (labels == 1)).item()
            nump += torch.sum((preds == 1)).item()

        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)
        train_acc2 = (TP+TN)/(TP+TN+FP+FN+ 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        F1 = 2 * precision * recall / (precision + recall + 1e-6)
        nump = nump / len(train_loader.dataset)


        if(epoch % 2 == 0):
            print("Train Loss: {}, Train Acc: {}".format(train_loss,train_acc))
            print("Train Acc2: {}".format(train_acc2))
            print("Train precision: {}, Train recall: {}".format(precision, recall))
            print("Train  F1: {}, positive rate: {}".format(F1, nump))
            test_acc, _, _ = test(model, test_loader, loss_func)
            if(best_val_acc < test_acc):
                best_val_acc = test_acc
                best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model

processor = DataProcessor()
train_datasets, test_datasets = processor.get_datasets(vocab_size=vocab_size, embedding_size=embedding_size, max_len=sentence_max_len)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

model = BiLSTMModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.BCELoss()
model = train(model, train_loader, test_loader, optimizer, loss_func, epochs)

_, preds, ids = test(model, test_loader, loss_func)
probs = pd.DataFrame({'id': ids, 'probability': preds})
probs.to_csv('./out/Twibot-20_predictions.csv', index=False)
