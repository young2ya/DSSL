import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.classification import Accuracy
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

def accuracy_(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

class Accuracy_(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def updata(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

def evaluation(loader, model, criterion):
    losses = AverageMeter()
    acc = Accuracy(task='multiclass', num_classes=7).cuda()
    # acc = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec = acc(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            # acc.update(prec.item(), inputs.size(0))
    return losses.avg, acc.compute()


def evaluation_hcae(loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs,_ = model(inputs)
            loss = criterion(outputs, targets)

            prec = accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            acc.update(prec.item(), inputs.size(0))
    return losses.avg, acc.avg

def Tsne(loader, model, ex, args):
    ### Extract latent features
    model.eval()
    with torch.no_grad():
        feature_list = []
        for data in loader:
            inputs, targets = data[0].cuda(), data[1].cuda()
            if args.model == 'wrn':
                outputs = model.conv1(inputs)
                outputs = model.block1(outputs)
                outputs = model.block2(outputs)
                outputs = model.block3(outputs)
                outputs = model.relu(model.bn1(outputs))
                outputs = F.avg_pool2d(outputs, 8)
            else:
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

