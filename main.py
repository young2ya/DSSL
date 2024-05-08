import os
import argparse

import random
import numpy as np
import pandas as pd
from copy import deepcopy
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import LambdaLR

from model import *
from train import *
from loader import *
from evaluation import *

parser = argparse.ArgumentParser(description='SSL Training')

parser.add_argument('--method', default='fixmatch', type=str)
parser.add_argument('--model', default='wrn')

parser.add_argument('--ex-epochs', '-e', default=5, type=int)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--npz', default=5, type=int)
parser.add_argument('--label-num', '-ln', default=10, type=int)
parser.add_argument('--train-iteration', '-ti', default=14, type=int)

parser.add_argument('--train-batch-size', default=32)
parser.add_argument('--batch-size', default=512)

parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--threshold', default=0.8, type=float)
parser.add_argument('--patience-limit', '-pl', default=25, type=int)

parser.add_argument('--seed', default=42, type=int, help='random seed')

parser.add_argument('--max-alpha', default=3)
parser.add_argument('--T1', default=10, type=int)
parser.add_argument('--T2', default=60, type=int)

parser.add_argument('--gpu', default='0')

parser.add_argument('--mode', default='client')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', default=65361)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
    data_path = 'D:\\SLRA_Bearing_data\\data_stft_64'
    path_list = [os.path.join(data_path, f_name) for f_name in os.listdir(data_path)]

    Test_acc = np.zeros((args.ex_epochs, 1))

    for i in range(args.ex_epochs):
        print(f'repeat experiment : {i + 1}')
        print('data loading .. ')

        # 1. Load dataset
        if args.method == 'cnn':
            trainset, testset, valset, _ = Data_Loader(path_list, args)
        elif args.method == 'pseudo':
            trainset, testset, valset, unlabelset = Data_Loader(path_list, args)
        elif args.method == 'hcae':
            trainset, testset, valset, unlabelset = Data_Loader(path_list, args)
        elif args.method == 'mixmatch':
            transform = transforms.Compose([Noise(), ToTensor()])
            trainset, testset, valset, unlabelset = Data_Loader_MixMatch(path_list, args, train_transform=transform)
        elif args.method == 'fixmatch':
            transform_w = transforms.Compose([Noise_w(), ToTensor()])
            transform_s = transforms.Compose([Noise_s(), ToTensor()])
            trainset, testset, valset, unlabelset = Data_Loader_FixMatch(path_list, args, transform_w, transform_s)

        # 2. Data Loader
        train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        if args.method != 'cnn':
            unlabel_loader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # 3.Create Model
        print('creating model ..')
        model = create_model(args)

        semi_criterion = SemiLoss()
        criterion = nn.CrossEntropyLoss()
        u_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_loss = 10**9
        patience_check = 0
        dur = []
        for epoch in range(args.start_epoch, args.epochs):
            print(f'\nEpoch: [{epoch + 1}|{args.epochs}]')
            t0 = time.time()

            if args.method == 'cnn':
                train_loss, train_acc = CNN_train(train_loader, model, optimizer, criterion)
            elif args.method == 'pseudo':
                train_loss, train_acc = Pseudo_train(train_loader, unlabel_loader, model, optimizer, criterion, epoch, args)
            elif args.method == 'hcae':
                train_loss, train_acc = HCAE_train(train_loader, unlabel_loader, model, optimizer, criterion, u_criterion, epoch, args)
            elif args.method == 'mixmatch':
                train_loss = MixMatch_trian(train_loader, unlabel_loader, model, optimizer, semi_criterion, epoch, args)
                _,train_acc = evaluation(train_loader, model, criterion)
            elif args.method == 'fixmatch':
                train_loss, train_acc = FixMatch_train(train_loader, unlabel_loader, model, optimizer, args)
                # _,train_acc = evaluation(train_loader, model, criterion)

            if args.method != 'hcae':
                val_loss, val_acc = evaluation(val_loader, model, criterion)
            else:
                val_loss, val_acc = evaluation_hcae(val_loader, model, criterion)

            if val_loss >= best_loss:
                patience_check += 1
                if patience_check >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(val_loss)
                patience_check = 0
                torch.save(model.state_dict(), f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}.pth')

            dur.append(time.time() - t0)

            print(f'Train loss : {train_loss:.4f}, Train acc : {train_acc:.4f}, Val loss : {val_loss:.4f}, val acc : {val_acc:.4f} -> {patience_check} / {np.mean(dur)}(s)')

        model.load_state_dict(torch.load(f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}.pth'))
        if args.method != 'hcae':
            test_loss, test_acc = evaluation(test_loader, model, criterion)
        else:
            test_loss, test_acc = evaluation_hcae(test_loader, model, criterion)

        Tsne(test_loader, model, i, args)

        print(f'Test acc : {test_acc:.4f}')

        Test_acc[i][0] = test_acc

    Test_acc = pd.DataFrame(Test_acc)
    Test_acc.to_csv(f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}_result.csv')
    print('end')

if __name__=='__main__':
    main()

# fixmatch -> lr 0.01, Adam, momentum(x),
# mixmatch -> lr 0.001, Adam, momentum(x),