import os
import random
import numpy as np
import pandas as pd
import argparse
from copy import deepcopy
import math
import shutil
import logging
#!
import time
from torch.optim.lr_scheduler import LambdaLR

import torch
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

# model
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
        x = self.features(x)
        x = x.view(-1, 128*8*8)
        x = self.classifier(x)
        return x


# data load
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
    data_dict = dict()
    label_dict = dict()
    for folder_path in folder_path_list:
        bearing_type = folder_path.split("\\")[-1]
        data_arr, label_arr = NPZ_Loader(folder_path=folder_path, each_num=each_data_num)

        data_dict[bearing_type] = data_arr
        label_dict[bearing_type] = label_arr

    return data_dict, label_dict

def Seed_Load(folder_path_list, args):
    data_dict = dict()
    label_dict = dict()
    for folder_path in folder_path_list:
        bearing_type = folder_path.split('\\')[-1]
        print(f'loading {bearing_type}..')
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

def Data_Loader_Pseudo(path_list, args):
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


def evaluation(loader, model, criterion):
    losses = AverageMeter()
    acc = Accuracy(task='multiclass', num_classes=7).cuda()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
    return losses.avg, acc.compute()

def Tsne(loader, model, ex, args):
    ### Extract latent features
    model.eval()
    with torch.no_grad():
        feature_list = []
        for data in loader:
            inputs, targets = data[0].cuda(), data[1].cuda()

            outputs = model.features(inputs)

            feature = outputs.detach().cpu().numpy()
            feature_list.append(feature)

        feature_mat = np.concatenate(feature_list, axis=0)
        feature_mat = feature_mat.reshape(feature_mat.shape[0], -1)
        label = loader.dataset.y_data

        N_index = np.where(label == 0)[0]
        SB_index = np.where(label == 1)[0]
        SI_index = np.where(label == 2)[0]
        SO_index = np.where(label == 3)[0]
        WB_index = np.where(label == 4)[0]
        WI_index = np.where(label == 5)[0]
        WO_index = np.where(label == 6)[0]

        ### Fit TSNE
        tsne = TSNE(n_components=2, n_iter=300)
        tsne_value = tsne.fit_transform(feature_mat)

        ### Plot the TNSE value
        plt.figure(figsize=(13, 9))
        plt.scatter(tsne_value[:, 0][N_index], tsne_value[:, 1][N_index], c='navy', s=60, label='N')
        plt.scatter(tsne_value[:, 0][SB_index], tsne_value[:, 1][SB_index], c='firebrick', s=60, label='SB')
        plt.scatter(tsne_value[:, 0][SI_index], tsne_value[:, 1][SI_index], c='green', s=60, label='SI')
        plt.scatter(tsne_value[:, 0][SO_index], tsne_value[:, 1][SO_index], c='indigo', s=60, label='SO')
        plt.scatter(tsne_value[:, 0][WB_index], tsne_value[:, 1][WB_index], c='darkviolet', s=60, label='WB')
        plt.scatter(tsne_value[:, 0][WI_index], tsne_value[:, 1][WI_index], c='gold', s=60, label='WI')
        plt.scatter(tsne_value[:, 0][WO_index], tsne_value[:, 1][WO_index], c='royalblue', s=60, label='WO')
        plt.legend(loc='upper right', fontsize='xx-large')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\TSNE_{args.method}_{args.label_num}_{ex+1}.png')

        return plt.close("all")


def Pseudo_train(train_loader, unlabel_loader, model, optimizer, criterion, epoch, args):
    losses = AverageMeter()
    acc = Accuracy(task='multiclass', num_classes=7).cuda()
    correct = 0
    total = 0

    if (epoch > args.T1) and (epoch < args.T2):
        alpha = args.max_alpha * (epoch - args.T1) / (args.T2 - args.T1)
    elif epoch >= args.T2:
        alpha = args.max_alpha
    else:
        alpha = 0

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

        inputs = torch.cat((inputs_x, inputs_u)).cuda()

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
        inputs_u = inputs_u.cuda()

        outputs_x = model(inputs_x)
        if alpha > 0:
            outputs_u = model(inputs_u)

            outputs_x = torch.softmax(outputs_x.detach(), dim=-1)
            outputs_u = torch.softmax(outputs_u.detach(), dim=-1)

            _, pred_u = torch.max(outputs_u.detach(), 1)

            loss = criterion(outputs_x, targets_x) + alpha * criterion(outputs_u, pred_u)
        else:
            outputs_x = torch.softmax(outputs_x.detach(), dim=-1)

            loss = criterion(outputs_x, targets_x)

        _, pred_x = torch.max(outputs_x, dim=-1)

        # total += targets_x.size(0)
        # correct += (pred_x == targets_x).sum().item()
        prec = acc(pred_x, targets_x)

        losses.update(loss.item(), targets_x.size(0))
        # acces.update(100*correct/total, targets_x.size(0))

        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return losses.avg, acc.compute(), model




# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser(description='SSL Training')
parser.add_argument('--method', default='pseudo', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--npz', default=5, type=int)
parser.add_argument('--label-num', '-ln', default=10, type=int)
parser.add_argument('--train-iteration', '-ti', default=14, type=int)

parser.add_argument('--ex-epochs', '-e', default=20, type=int)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('--train-batch-size', default=16)
parser.add_argument('--batch-size', default=512)
parser.add_argument('--threshold', default=0.8, type=float)


parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)

parser.add_argument('--max-alpha', default=1)
parser.add_argument('--T1', default=10, type=int)
parser.add_argument('--T2', default=50, type=int)


parser.add_argument('--patience-limit', '-pl', default=20, type=int)
parser.add_argument('--gpu', default='0')
# !
parser.add_argument('--nesterov', action='store_true', default=True, help='use nexterov momentum')
parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
parser.add_argument('--total-steps', default=2**20, type=int, help='number of total steps to run')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA model')
parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')

parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='seed_data.txt', help='directory to output the result')

parser.add_argument('--mode', default='client')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', default=65361)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

logger = logging.getLogger(__name__)

def main():
    data_path = 'D:\\SLRA_Bearing_data\\data_stft_64'
    path_list = [os.path.join(data_path, f_name) for f_name in os.listdir(data_path)]

    Test_acc = np.zeros((args.ex_epochs,1))
    for i in range(args.ex_epochs):
        print(f'repeat experiment : {i + 1}')

        print('creating model ..')
        model = CNN(num_classes=7).cuda()
        logger.info('Total param: {:.2f}M'.format(sum(p.numel() for p in model.parameters())/1e6))

        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO)

        logger.info(dict(args._get_kwargs()))

        # if args.seed is not None:
        #     set_seed(args)

        print('data loading ..')
        trainset, testset, valset, unlabelset = Data_Loader_Pseudo(path_list, args)

        train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        unlabel_loader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9 )
        criterion = nn.CrossEntropyLoss()

        # if args.use_ema:
        #     ema_model = ModelEMA(args, model, args.ema_decay)

        best_acc = 0
        patience_check = 0
        test_accs = []
        for epoch in range(args.start_epoch, args.epochs):
            print(f'\nEpoch: [{epoch + 1}|{1000}]   P_{patience_check}  {i+1}')

            train_loss, train_acc, model = Pseudo_train(train_loader, unlabel_loader, model, optimizer, criterion, epoch, args)
            logger.info('train_loss : {:.4f}'.format(train_loss))
            logger.info('train_acc : {:.4f}'.format(train_acc))

            # if args.use_ema:
            #     test_model = ema_model.ema
            # else:
            #     test_model = model

            val_loss, val_acc = evaluation(val_loader, model, criterion)
            logger.info('eval_loss : {:.2f}'.format(val_loss))
            logger.info('eval_acc : {:.2f}'.format(val_acc))

            if val_acc <= best_acc:
                patience_check += 1
                if patience_check > args.patience_limit:
                    break
            else:
                best_acc = deepcopy(val_acc)
                patience_check = 0
                torch.save(model.state_dict(), f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}.pth')

            test_accs.append(val_acc.cpu())
            logger.info('Best acc: {:.2f}'.format(best_acc))
            logger.info('Mean acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

        model.load_state_dict(torch.load(f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}.pth'))
        test_loss, test_acc = evaluation(test_loader, model, criterion)

        Tsne(test_loader, model, i, args)

        logger.info('Test acc: {:.2f}'.format(test_acc))

        Test_acc[i][0] = test_acc

    Test_acc = pd.DataFrame(Test_acc)
    Test_acc.to_csv(f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}_result.csv')

    print('end')

if __name__ == '__main__':
    main()

