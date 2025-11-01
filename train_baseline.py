import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
import numpy as np
import hashlib
import pickle
import time
from tqdm import tqdm
import sys

# 输入参数
path = sys.argv[1]
BATCH_SIZE = 64

# HPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHP
_denseSize1 = 8192
_denseSize2 = 2048
_lr = 0.00001
# HPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHPHP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#切片，获取第一个'_'到最后一个'_'之间的字符串
tag = sys.argv[1][sys.argv[1].find('_') + 1 : sys.argv[1].rfind('_')]

import os

os.environ['CUDA_VISIBLE_DEVICE']='0'
#检查当前目录是否存在目标文件夹，没有则创建文件夹
if not os.path.exists('model'):
    os.system('mkdir model')
if not os.path.exists('preprocessed_data'):
    os.system('mkdir preprocessed_data')

#检查是否存在某个训练好的模型(.torchsave)，若存在直接退出
fileprefix = "model/model_{}_{}_{}_{}".format(tag, _denseSize1, _denseSize2, _lr)
if os.path.exists(fileprefix + ".torchsave"):
    print("Skip. Already trained Model: " + fileprefix + ".torchsave")
    sys.exit()

#创建一个日志文件(.log)
logfile = open(fileprefix + ".log", 'w')

#日志文件写入函数
def log(logstr):
    print(logstr)
    logfile.write(logstr + '\n')

# Read data
from glob import glob
original_data = glob(os.path.join(path, '*/*'))

log("Original data:" + str(len(original_data))) #将原始文件总数写入日志

# numCluster = len(original_data) // 100
numCluster = sum(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path))
print("numCluster:",numCluster)

train_data = []
test_data = []

# random.seed(1)
# for i in range(numCluster):
#     name = glob(os.path.join(path,'{}/*'.format(i)))
#     # windows下额外处理
#     name = [path.replace('\\', '/') for path in name]
#     random.shuffle(name)
#     n = len(name)
#     # train_data += name[:n // 10]
#     # test_data += name[n // 10:]
#     train_data += name[:int(n * 0.7)]  # 70% 训练数据
#     test_data += name[int(n * 0.7):]   # 30% 测试数据

random.seed(1)
all_data = glob(os.path.join(path, '*/*'))
all_data = [path.replace('\\', '/') for path in all_data]
random.shuffle(all_data)  # 随机打乱数据顺序
split_point = int(len(all_data) * 0.8)  # 计算训练集和测试集的分割点
train_data = all_data[:split_point]  # 前70%作为训练集
test_data = all_data[split_point:]   # 后30%作为测试集

log("Train data: " + str(len(train_data))) # 训练集文件总数写入日志
log("Test data: " + str(len(test_data)))   #测试集文件总数写入日志


#输入文件名列表和采样率，生成一个16字符的哈希值
def files_to_hash(files, sampling):
    m = hashlib.sha256()
    m.update(str(sampling).encode())
    for fn in sorted(files):
        m.update(fn.encode())

    return m.hexdigest()[:16] # Use the first 16 chars. It's too long

# Use it for large datasets
class RuntimeLoader:
    def __init__(self, files):
        self.dataset = files.copy()
        random.seed(1)
        random.shuffle(self.dataset)
        self.num = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        num = self.num
        nowlen = min(len(self.dataset) - num, BATCH_SIZE)
        if nowlen > 0:
            self.num += nowlen
            ret1 = [[] for i in range(nowlen)]
            ret2 = []
            for i in range(nowlen):
                with open(self.dataset[num + i], 'rb') as f:
                    data = f.read()
                data = [int(d) for d in data]
                ret1[i] = data
                ret2.append(int(self.dataset[num + i][self.dataset[num + i].rfind('/') + 1:self.dataset[num + i].rfind('_')]))
            ret = [(torch.tensor(ret1) - 128) / 128.0, torch.tensor(ret2)]

            for i in range(len(ret)):
                ret[i] = ret[i].cpu()
            return ret
        else:
            raise StopIteration

# Use it when the data is small enough to store in the main memory
class Loader:
    def __init__(self, files, sampling=1.0):
        picklename = "preprocessed_data/" + files_to_hash(files, sampling) + ".data"
        if os.path.exists(picklename):
            with open(picklename, 'rb') as f:
                self.alldata = pickle.load(f)
                self.alllen = self.alldata[2]
        else:
            self.dataset = files.copy()
            random.seed(1)
            random.shuffle(self.dataset)
            if sampling < 1.0:
                self.dataset = self.dataset[:int(len(self.dataset) * sampling)]
            self.alldata = self.load_data()
            self.alllen = self.alldata[2]
            with open(picklename, 'wb') as f:
                pickle.dump(self.alldata, f)
        self.num = 0

    def load_data(self):
        alllen = len(self.dataset)
        ret1 = [[] for i in range(alllen)]
        ret2 = []
        for i in range(alllen):
            with open(self.dataset[i], 'rb') as f:
                data = f.read()
            data = [int(d) for d in data]
            ret1[i] = data
            fn = self.dataset[i]
            ret2.append(int(fn[fn.rfind('/') + 1:fn.rfind('_')]))
        return ((torch.tensor(ret1) - 128) / 128.0), torch.tensor(ret2), alllen

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        nowlen = min(self.alllen - num, BATCH_SIZE)

        if nowlen > 0:
            self.num += nowlen
            ret1 = self.alldata[0][num:num + nowlen]
            ret2 = self.alldata[1][num:num + nowlen]
            ret = [ret1, ret2]

            for i in range(len(ret)):
                ret[i] = ret[i].cpu()
            return ret
        else:
            raise StopIteration


train_data_tensor = list(Loader(train_data, 1.0))
#test_data_tensor = list(RuntimeLoader(test_data))

log("Tensor conversion done") # 张量转化完成


def test(model, test_loader, epoch, print_progress=False):
    model.eval()
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    total = 0

    # 从 test_loader 的 alldata 中获取所有可能的类别标签
    all_classes = set(test_loader.alldata[1].tolist())

    # 跟踪实际出现的标签
    actual_classes = set()

    # 初始化每个类别的TP、FP、FN为0，只初始化所有可能的类别
    total_tp = {cls: 0 for cls in all_classes}
    total_fp = {cls: 0 for cls in all_classes}
    total_fn = {cls: 0 for cls in all_classes}

    # 计算每个类别的支持
    support = {cls: 0 for cls in all_classes}

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            prob, label = output.topk(5, 1, True, True)

            expanded = target.view(target.size(0), -1).expand_as(label)
            compare = label.eq(expanded).float()

            total += len(data)
            correct_1 += int(compare[:, :1].sum())
            correct_5 += int(compare[:, :5].sum())

            # 计算每个类别的TP、FP、FN
            pred_labels = output.argmax(dim=1)
            for t, p in zip(target, pred_labels):
                t = t.item()
                p = p.item()
                actual_classes.add(t)  # 跟踪实际出现的标签

                # 确保标签存在于字典中
                if t not in total_fn:
                    total_fn[t] = 0
                if p not in total_fp:
                    total_fp[p] = 0

                total_fn[t] += 1  # 每个样本的真正类别都会被计数
                support[t] += 1  # 计算支持
                if p == t:
                    total_tp[t] += 1
                else:
                    total_fp[p] += 1

            # 计算每个类别的Precision和Recall
            precision_per_class = {
                cls: total_tp.get(cls, 0) / (total_tp.get(cls, 0) + total_fp.get(cls, 0))
                if (total_tp.get(cls, 0) + total_fp.get(cls, 0)) > 0 else 0
                for cls in actual_classes
            }
            recall_per_class = {
                cls: total_tp.get(cls, 0) / (total_tp.get(cls, 0) + total_fn.get(cls, 0))
                if (total_tp.get(cls, 0) + total_fn.get(cls, 0)) > 0 else 0
                for cls in actual_classes
            }
            # 计算加权Precision和Recall
            weighted_precision = sum(
                support.get(cls, 0) * precision_per_class.get(cls, 0) for cls in actual_classes) / sum(support.values())
            weighted_recall = sum(
                support.get(cls, 0) * recall_per_class.get(cls, 0) for cls in actual_classes) / sum(support.values())

            # 计算F1 Score
            def f1_score(precision, recall):
                return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            weighted_f1 = f1_score(weighted_precision, weighted_recall)

            print('total:{}'.format(total))
            if total > 0:
                log('Test Epoch: {}, Top 1 accuracy: {}/{} ({:.2f}%), Top 5 accuracy: {}/{} ({:.2f}%), Weighted Precision: {:.4f}, Weighted Recall: {:.4f}, Weighted F1 Score: {:.4f}'.format(
                    epoch, correct_1, total, 100. * correct_1 / total,
                    correct_5, total, 100. * correct_5 / total, weighted_precision, weighted_recall, weighted_f1))
            return (test_loss / total, correct_1, correct_5)


def do_test(sampling, print_progress=False):
    if sampling == 1.0:
        # it = RuntimeLoader(test_data)
        it = Loader(test_data, sampling)
    else:
        it = Loader(test_data, sampling)

    test(hidden_model, it, epoch, print_progress)

def do_eval(sampling):
    it = Loader(train_data, sampling)
    test(hidden_model, it, epoch)

class RevisedNetwork(torch.nn.Module):
    def __init__(self):
        super(RevisedNetwork, self).__init__()

        self.conv_layers = []
        self.layers = []

        self.conv_layers.append(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)) 
        self.conv_layers.append(nn.ReLU()) 
        self.conv_layers.append(nn.BatchNorm1d(8))
        self.conv_layers.append(nn.MaxPool1d(2)) 

        # self.conv_layers.append(nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True))
        # self.conv_layers.append(nn.ReLU())
        # self.conv_layers.append(nn.BatchNorm1d(16))
        # self.conv_layers.append(nn.MaxPool1d(2))
        #
        # self.conv_layers.append(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True))
        # self.conv_layers.append(nn.ReLU())
        # self.conv_layers.append(nn.BatchNorm1d(32))
        # self.conv_layers.append(nn.MaxPool1d(2))

        self.layers.append(nn.Linear(8192 * 4, _denseSize1))
        self.layers.append(nn.ReLU()) 
        self.layers.append(nn.Dropout(p=0.5))


        last_denseSize = _denseSize1
        if _denseSize2 > 0:
            self.layers.append(nn.Linear(_denseSize1, _denseSize2))
            self.layers.append(nn.ReLU()) 
            self.layers.append(nn.Dropout(p=0.5))
            last_denseSize = _denseSize2

        self.fc = nn.Linear(last_denseSize, numCluster, bias=False)

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.layers = nn.ModuleList(self.layers)



    def forward(self, x):
        x = x.unsqueeze(dim=1)
        for l in self.conv_layers:
            x = l(x)

        x = x.view(x.shape[0], -1)
        for l in self.layers:
            x = l(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
        

hidden_model = RevisedNetwork()
device = "cpu"
hidden_model= hidden_model.to(device)
optimizer = optim.Adam(hidden_model.parameters(), lr = _lr, weight_decay=1e-4)

loss = []
prevtime = time.time()
prevloss = []
for epoch in range(1, 351):
    train_loss = 0
    # data是一个文件转化成的张量，target是文件的索引(猜测)，batch_idx是循环变量
    for batch_idx, (data, target) in enumerate(train_data_tensor):
        optimizer.zero_grad()
        outputs = hidden_model(data)
        loss = F.nll_loss(outputs, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_data_tensor)
    log('Epoch: {}\tLoss: {:.6f}\tTime: {}'.format(
        epoch, train_loss, time.time()-prevtime))
    prevloss.append(train_loss)
    if len(prevloss) >= 10:
        mx = max(prevloss[-10:])
        mi = min(prevloss[-10:])
        if (mx - mi) / mi < 0.05:
            break

    prevtime = time.time()

    if epoch % 10 == 0:
        do_test(0.01)
        do_eval(0.1)
        torch.save(hidden_model.state_dict(), fileprefix + ".cp.torchsave")

torch.save(hidden_model.state_dict(), fileprefix + ".torchsave")
do_test(1.0, True)
logfile.close()
