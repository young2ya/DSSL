import os
import argparse

import numpy as np
import pandas as pd
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # 64
            nn.Conv2d(1,32,5,padding=2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            #32
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            #16
            nn.Conv2d(64, 128, 5, padding=2),
            nn.MaxPool2d(2,2),
            nn.ReLU())
            #8
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, num_classes))

    def forward(self, x):
        logit = self.features(x)
        x = logit.view(-1, 128*8*8)
        x = self.classifier(x)
        return logit, x

def NPZ_Loader(folder_path, each_num):
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".npz")]
    file_names = random.sample(file_names, each_num)

    npy_list = []
    label_list = []
    for file_name in file_names:
        npz = np.load(file_name)
        x = npz['x']
        y = npz['y']

        npy_list.append(x)
        label_list.append(y)
    npy_arr = np.vstack(npy_list)
    label_arr = np.hstack(label_list)

    return npy_arr, label_arr

def Seed_NPZ_Loader(folder_path, args):
    random_seed = random.Random(args.seed)
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.npz')]
    file_names = random_seed.sample(file_names, 1)

    npy_list = []
    label_list = []
    for file_name in file_names:
        npz = np.load(file_name)
        x = npz['x']
        y = npz['y']

        npy_list.append(x)
        label_list.append(y)
    npy_arr = np.vstack(npy_list)
    label_arr = np.hstack(label_list)

    return npy_arr, label_arr

def Load(folder_path_list, each_data_num):
    arr_dict = dict()
    label_dict = dict()
    for folder_path in folder_path_list:
        bearing_type = folder_path.split("\\")[-1]
        print(f'loading {bearing_type}..')
        npy_arr, label_arr = NPZ_Loader(folder_path=folder_path, each_num=each_data_num)

        arr_dict[bearing_type] = npy_arr
        label_dict[bearing_type] = label_arr

    return arr_dict, label_dict

def Seed_Load(folder_path_list, args):
    data_dict = dict()
    label_dict = dict()
    for folder_path in folder_path_list:
        bearing_type = folder_path.split('\\')[-1]
        train_data, train_label = Seed_NPZ_Loader(folder_path, args)

        data_dict[bearing_type] = train_data
        label_dict[bearing_type] = train_label

    return data_dict, label_dict

def Slice(data_dict, label_dict, eval_data_num):
    test_data, val_data, unlabel_data, test_label, val_label, unlabel_label = [{} for i in range(6)]

    test_num = eval_data_num
    val_num = test_num + eval_data_num
    unlabel_num = val_num + eval_data_num

    for key, value in zip(data_dict.keys(), data_dict.values()):
        test_data[key] = value[:test_num]
        val_data[key] = value[test_num:val_num]
        unlabel_data[key] = value[val_num:unlabel_num]

    for key, value in zip(label_dict.keys(), label_dict.values()):
        test_label[key] = value[:test_num]
        val_label[key] = value[test_num:val_num]
        unlabel_label[key] = value[val_num:unlabel_num]

    return test_data, val_data, unlabel_data, test_label, val_label, unlabel_label

def Seed_Slice(data_dict, label_dict, args):  # slice가 아니라 seed고정 random sampling해야함
    random_seed = random.Random(args.seed)
    train_data, train_label = [{} for i in range(2)]

    train_idx = random_seed.sample(range(1000), args.label_num)

    for key, value in zip(data_dict.keys(), data_dict.values()):
        train_data[key] = value[train_idx]

    for key, value in zip(label_dict.keys(), label_dict.values()):
        train_label[key] = value[train_idx]

    return train_data, train_label

def Concat_data(X_dict, y_dict):
    X_concat = np.concatenate(list(X_dict.values()), axis=0)
    y_concat = np.concatenate(list(y_dict.values()), axis=0)

    X_concat = X_concat.reshape(X_concat.shape + (1,))

    return X_concat, y_concat

class MyDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = torch.FloatTensor(x_data).permute(0,3,2,1)
        self.y_data = torch.LongTensor(y_data)
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        signal, target = self.x_data[index], self.y_data[index]
        if self.transform is not None:
            signal = self.transform(signal)
        return signal, target

    def __len__(self):
        return self.len

def Data_Loader(path_list, args):
    random_data, random_label = Load(path_list, args.npz)
    seed_data, seed_label = Seed_Load(path_list, args)

    shuffle_data = dict()
    for key, value in zip(random_data.keys(), random_data.values()):
        np.random.shuffle(value)
        shuffle_data[key] = value

    test_data, val_data, unlabel_data, test_label, val_label, unlabel_label = Slice(shuffle_data, random_label, 1000)
    train_data, train_label = Seed_Slice(seed_data, seed_label, args)

    X_train, y_train = Concat_data(train_data, train_label)
    X_test, y_test = Concat_data(test_data, test_label)
    X_val, y_val = Concat_data(val_data, val_label)
    X_unlabel, y_unlabel = Concat_data(unlabel_data, unlabel_label)

    trainset = MyDataset(X_train, y_train)
    testset = MyDataset(X_test, y_test)
    valset = MyDataset(X_val, y_val)
    unlabelset = MyDataset(X_unlabel, y_unlabel)

    return trainset, testset, valset, unlabelset
############################### MetMatch train 짜는 중...
def train(train_loader, model):

    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabel_loader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(train_loader)
            inputs_x, targets_x = next(labeled_train_iter)
        try:
            inputs_u,_ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            inputs_u,_ = next(unlabeled_train_iter)

    inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
    inputs_u = inputs_u.cuda()

    latent_x, outputs_x = model(inputs_x)
    latent_u, outputs_u = model(inputs_u)

    softmax_outputs_x = torch.softmax(outputs_x.detach(), dim=-1)
    softmax_outputs_u = torch.softmax(outputs_u.detach(), dim=-1)   # unlabeled data의 class별 확률정보 (pseudo label)

    a = targets_x[0]
    b = softmax_outputs_u[0][a]
    c = torch.cat([softmax_outputs_u[0][0:a], softmax_outputs_u[0][a+1:]])
    d = max(c)
    e = b-d

    for i in latent_x:
        for r in latent_u:
            distance = (i - r).pow(2).sum(3).sqrt()



    a = nn.CrossEntropyLoss()
    loss = a(softmax_outputs_x, targets_x)
    
def triplet_loss(anchor, pred_prob, alpha=0.3):

    return
###############################
parser = argparse.ArgumentParser(description='MetMatch')

parser.add_argument('--npz', default=5, type=int)
parser.add_argument('--label-num', '-ln', default=10, type=int)

parser.add_argument('--train-iteration', '-ti', default=14, type=int)
parser.add_argument('--train-batch-size', default=32)
parser.add_argument('--batch-size', default=512)

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--mode', default='client')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', default=65361)

args = parser.parse_args()


data_path = 'D:\\SLRA_Bearing_data\\data_stft_64'
path_list = [os.path.join(data_path, f_name) for f_name in os.listdir(data_path)]

trainset, testset, valset, unlabelset = Data_Loader(path_list, args)

train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
unlabel_loader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, drop_last=True)

model = CNN(num_classes=7).cuda()






####04.18 labeled data seed 고정되어 도출되는지 확인 중 -> 같음!
