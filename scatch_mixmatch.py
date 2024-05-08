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

class MyDataset_MixMatch(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = torch.FloatTensor(x_data).permute(0,3,2,1)
        self.y_data = torch.LongTensor(y_data)
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        signal_1, signal_2, target = self.x_data[index], self.x_data[index], self.y_data[index]

        if self.transform is not None:
            segnal_1 = self.transform(signal_1)
            signal_2 = self.transform(signal_2)

        signal = signal_1, signal_2
        return signal, target

    def __len__(self):
        return self.len

def Data_Loader_MixMatch(path_list, args, train_transform = None):
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

    trainset = MyDataset(X_train, y_train, transform=train_transform)
    testset = MyDataset(X_test, y_test)
    valset = MyDataset(X_val, y_val)
    unlabelset = MyDataset_MixMatch(X_unlabel, y_unlabel, transform=train_transform)

    return trainset, testset, valset, unlabelset

class Noise(object):
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.rand(c, h, w)    #0~1
        return x

class ToTensor(object):
    def __call__(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x


def MixMatch_trian(train_loader, unlabel_loader, model, optimizer, criterion, epoch, args):
    losses = AverageMeter()

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
            (inputs_u, inputs_u2),_ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            (inputs_u, inputs_u2),_ = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 7).scatter_(1, targets_x.view(-1,1).long(),1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        # Compute guessed labels of unlabeled samples
        with torch.no_grad():
            outputs = model(inputs_u)
            outputs_2 = model(inputs_u2)

            p = (torch.softmax(outputs, dim=1) + torch.softmax(outputs_2, dim=1)) / 2
            pt = p**(1/args.T)

            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # MixUp
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1-l) * input_b
        mixed_target = l * target_a + (1-l) * target_b

        # interleave labeled  and unlabeled samples between to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch, batch_idx, args)
        loss = Lx + w * Lu

        losses.update(loss.item(), inputs_x.size(0))

        loss.backward()
        optimizer.step()
        # if args.use_ema:
        #     ema_model.update(model)
        optimizer.zero_grad()

        return losses.avg


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, batch_idx, args):
        probs_u = torch.softmax(outputs_u, dim=1)
        epochs = epoch + batch_idx/args.train_iteration

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epochs, args.epochs)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


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




parser = argparse.ArgumentParser(description='SSL Training')

parser.add_argument('--method', default='mixmatch', type=str)
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

parser.add_argument('--max-alpha', default=3)
parser.add_argument('--T1', default=10, type=int)
parser.add_argument('--T2', default=60, type=int)


parser.add_argument('--patience-limit', '-pl', default=25, type=int)
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

    Test_acc = np.zeros((args.ex_epochs, 1))

    for i in range(args.ex_epochs):
        print(f'repeat experimet : {i + 1}')

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

        transform = transforms.Compose([Noise(), ToTensor()])
        trainset, testset, valset, unlabelset = Data_Loader_MixMatch(path_list, args, train_transform=transform)

        train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        unlabel_loader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        semi_criterion = SemiLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # if args.use_ema:
        #     ema_model = ModelEMA(args, model, args.ema_decay)

        args.start_epoch = 0
        best_acc = 0
        patience_check = 0
        test_accs = []
        for epoch in range(args.start_epoch, args.epochs):
            print(f'\nEpoch: [{epoch + 1}|{1000}]   P_{patience_check}  {i+1}')

            MixMatch_trian(train_loader, unlabel_loader, model, optimizer, semi_criterion, epoch, args)
            _, train_acc = evaluation(train_loader, model, criterion)
            logger.info('train_acc : {:.4f}'.format(train_acc))

            # if args.use_ema:
            #     test_model = ema_model.ema
            # else:
            #     test_model = model

            val_loss, val_acc = evaluation(val_loader, model, criterion)
            logger.info('eval_loss : {:.4f}'.format(val_loss))
            logger.info('eval_acc : {:.4f}'.format(val_acc))

            if val_acc <= best_acc:
                patience_check += 1
                if patience_check > args.patience_limit:
                    break
            else:
                best_acc = deepcopy(val_acc)
                patience_check = 0
                torch.save(model.state_dict(), f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}.pth')

            test_accs.append(val_acc.cpu())
            logger.info('Best acc: {:.4f}'.format(best_acc))
            logger.info('Mean acc: {:.4f}\n'.format(np.mean(test_accs[-20:])))

        model.load_state_dict(
            torch.load(f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}.pth'))

        test_loss, test_acc = evaluation(test_loader, model, criterion)

        Tsne(test_loader, model, i, args)

        logger.info('Test acc: {:.4f}'.format(test_acc))

        Test_acc[i][0] = test_acc

    Test_acc = pd.DataFrame(Test_acc)
    Test_acc.to_csv(
        f'D:\\PycharmProjects\\DeepSSL\\result\\{args.method}\\{args.method}_{args.label_num}_result.csv')
    print('end')

if __name__ == '__main__':
    main()



