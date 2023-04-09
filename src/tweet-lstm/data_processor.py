# -*- coding: utf-8 -*-
"""
April 2023
Updated code by Madelyn Scandlen
Original code by Shangbin Feng et al. found at https://github.com/LuoUndergradXJTU/TwiBot-22/blob/master/src/Wei/Twibot-20/data_processor.py
Inspired by Feng Wei & Uyen Trang Nguyen
"""
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import sklearn

glove_path = './datasets/glove.twitter.27B.25d.txt'
torch.manual_seed(100)
class DataProcessor(object):
    def read_text(self,is_train_data):
        datas = []
        ids = []
        labels = []

        # json_file = pd.read_json('./datasets/Twibot-20/Twibot-20.json')
        json_file = pd.read_json('./datasets/Twibot-20/sample/Twibot-20.json')
        if (is_train_data):
            for line in json_file.itertuples():
                if (getattr(line, 'split') == "train") & (getattr(line, 'label') == "human"):
                    datas.append(getattr(line, 'text'))
                    ids.append(getattr(line, 'id'))
                    labels.append([1,0])
                elif (getattr(line, 'split') == "train") & (getattr(line, 'label') == "bot"):
                    datas.append(getattr(line, 'text'))
                    ids.append(getattr(line, 'id'))
                    labels.append([0,1])


        else:
            for line in json_file.itertuples():
                if (getattr(line, 'split') == "test") & (getattr(line, 'label') == "human"):
                    datas.append(getattr(line, 'text'))
                    ids.append(getattr(line, 'id'))
                    labels.append([1,0])
                elif (getattr(line, 'split') == "test") & (getattr(line, 'label') == "bot"):
                    datas.append(getattr(line, 'text'))
                    ids.append(getattr(line, 'id'))
                    labels.append([0,1])

        return datas, labels, ids

    def word_count(self, datas):
        dic = {}
        for data in datas:
            data_list = data.split()
            for word in data_list:
                if(word in dic):
                    dic[word] += 1
                else:
                    dic[word] = 1
        word_count_sorted = sorted(dic.items(), key=lambda item:item[1], reverse=True)
        return  word_count_sorted

    def word_index(self, datas, vocab_size, special_tokens):
        word_count_sorted = self.word_count(datas)
        word2index = {t:i for i,t in enumerate(special_tokens)}

        vocab_size = min(len(word_count_sorted), vocab_size)
        for i in range(vocab_size):
            word = word_count_sorted[i][0]
            word2index[word] = i + len(special_tokens)

        return word2index, vocab_size

    def get_datasets(self, vocab_size, embedding_size, max_len):

        special_tokens = ['<unk>', '<pad>', '<sep>', '<user>', '<hashtag>', '<url>', '<number>', '<repeat>', 'rt']

        train_datas, train_labels, train_ids = self.read_text(is_train_data = True)

        word2index, vocab_size = self.word_index(train_datas, vocab_size, special_tokens)

        test_datas, test_labels, test_ids = self.read_text(is_train_data = False)

        train_features = []
        embedding_index = dict()

        with open(glove_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:])
                embedding_index[word] = coefs
                line = f.readline()
        embedding_matrix = np.zeros((vocab_size + len(special_tokens), embedding_size))
        for word, i in word2index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


        for data in train_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower()
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"])
                if(len(feature)==max_len):
                    break
            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            train_features.append(feature)

        test_features = []
        for data in test_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                # word = word.lower()
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"])
                if(len(feature)==max_len):
                    break
            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            test_features.append(feature)

        train_features = torch.LongTensor(train_features)
        train_labels = torch.FloatTensor(train_labels)
        train_ids = torch.LongTensor(train_ids)

        test_features = torch.LongTensor(test_features)
        test_labels = torch.FloatTensor(test_labels)
        test_ids = torch.LongTensor(test_ids)

        embed = nn.Embedding(vocab_size + len(special_tokens), embedding_size)
        embed.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix))
        embed.weight.requires_grad = False
        train_features = embed(train_features)
        test_features = embed(test_features)

        train_features = torch.FloatTensor(train_features)
        test_features = torch.FloatTensor(test_features)

        train_features = train_features.float()
        test_features = test_features.float()

        train_features = Variable(train_features, requires_grad=False)
        train_datasets = torch.utils.data.TensorDataset(train_features, train_labels, train_ids)

        test_features = Variable(test_features, requires_grad=False)
        test_datasets = torch.utils.data.TensorDataset(test_features, test_labels, test_ids)
        return train_datasets, test_datasets
