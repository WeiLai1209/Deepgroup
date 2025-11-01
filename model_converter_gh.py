# 导入PyTorch库，它是用于深度学习的开源库
import torch

# 导入PyTorch的神经网络模块
import torch.nn as nn

# 导入PyTorch的神经网络功能函数模块
import torch.nn.functional as F

# 导入PyTorch的优化器模块，用于训练模型
import torch.optim as optim

# 从torchvision库中导入常用的数据集和图像转换工具
from torchvision import datasets, transforms

# 导入Python的random库，用于生成随机数
import random

# 导入NumPy库，用于数值计算
import numpy as np


# 导入hashlib库，用于加密哈希
import hashlib

# 导入pickle库，用于序列化和反序列化Python对象
import pickle

# 导入time库，用于处理时间相关的操作
import time

# 从glob库中导入glob函数，用于查找匹配特定模式的文件路径
from glob import glob

# 导入os和sys库，用于操作系统相关的任务
import os, sys

# 从collections库中导入defaultdict，它是一个字典子类，允许为不存在的键提供默认值
from collections import defaultdict

# 导入tqdm库，用于显示循环的进度条
from tqdm import tqdm

# 定义常量BATCH_SIZE，表示每个批次的大小为128
BATCH_SIZE = 128

# 定义变量numCluster，表示簇的数量，初始化为20000
numCluster = 20000

# 定义变量typeCluster，表示簇的类型，初始化为"large"
typeCluster = "large"

# 从命令行参数中获取文件名，并赋值给变量filename
filename = sys.argv[1]

# 获取filename的基本名称（不带路径），并赋值给变量bn
bn = os.path.basename(filename)

# 移除bn中的"model_hash_"字符串
bn = bn.replace("model_hash_", "")

# 移除bn中的".torchsave"字符串
bn = bn.replace(".torchsave", "")

# 将bn按'_'分割成列表，并赋值给变量modelinfo
modelinfo = bn.split('_')

# 检查modelinfo的长度是否不等于6，如果不等于6则打印modelinfo和错误信息，并退出程序
if len(modelinfo) != 6:

    print(modelinfo)

    print("Fail to extract model info!")

    sys.exit()

# 从modelinfo中提取标签，并赋值给变量tag
tag = modelinfo[0]

# 从modelinfo中提取哈希大小，并转换为整数，赋值给变量_hashSize
_hashSize = int(modelinfo[1])

# 从modelinfo中提取第一个密集层的大小，并转换为整数，赋值给变量_denseSize1
_denseSize1 = int(modelinfo[2])

# 从modelinfo中提取第二个密集层的大小，并转换为整数，赋值给变量_denseSize2
_denseSize2 = int(modelinfo[3])

# 从modelinfo中提取选择哪个密集层的索引，并转换为整数，赋值给变量_which_dense
_which_dense = int(modelinfo[4])

# 检查命令行参数的数量，如果大于等于3，则使用前两个参数组合成hashdict_filename

#python model_converter_gh.py D:\PythonProject\deepsketch-fast2022-main\training\model\model_hash_tag_128_8192_2048_1_5e-05.torchsave D:\PythonProject\deepsketch-fast2022-main\training\data_tag_pre
# 否则，将hashdict_filename设置为"hashdict.txt"

if len(sys.argv) >= 3:

    hashdict_filename = sys.argv[1] + "." + sys.argv[2]

else:

    hashdict_filename = "hashdict.txt"


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

#random.seed(1)
#torch.cuda.manual_seed(1)

# 读取数据
from glob import glob
path = sys.argv[2]  # 从命令行参数中获取数据路径

original_data = glob(os.path.join(path, '*/*'))  # 使用glob函数获取指定路径下的所有文件

# 计算簇的数量，即数据集大小除以100
# numCluster = len(original_data) // 100
numCluster = sum(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path))
print("numCluster:",numCluster)

# 定义GreedyHashLoss类，继承自torch.nn.Module
class GreedyHashLoss(torch.nn.Module):
    def __init__(self, bit, alpha=1):
        super(GreedyHashLoss, self).__init__()
        # 定义一个线性层，输入维度为bit，输出维度为numCluster，不使用偏置项，并将模型发送到指定设备
        self.fc = torch.nn.Linear(bit, numCluster, bias=False).to(device)
        # 初始化交叉熵损失函数，并发送到指定设备
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        # 设置alpha参数
        self.alpha = alpha

    def forward(self, outputs, y, feature):
        # 计算交叉熵损失
        loss1 = self.criterion(outputs, y)
        # 计算正则化项损失，即特征绝对值的三次方再取绝对值的均值
        loss2 = self.alpha * (feature.abs() - 1).pow(3).abs().mean()
        # 返回总损失
        return loss1 + loss2

        # 定义Hash类，继承自torch.autograd.Function，用于自定义反向传播
    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # 保存输入用于反向传播（这里注释掉了）
            # ctx.save_for_backward(input)
            # 返回输入的符号函数结果
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # 获取保存的输入（这里注释掉了）
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            # 直接返回grad_output作为梯度，因为符号函数的导数是0或-1，这里简化为1
            return grad_output

        # 定义RevisedNetwork类，继承自torch.nn.Module
class RevisedNetwork(torch.nn.Module):
    def __init__(self):
        super(RevisedNetwork, self).__init__()

        self.conv_layers = []  # 存储卷积层的列表
        self.layers = []  # 存储其他层的列表（这里未使用）

        # 复制自train_baseline.py的网络结构
        # 添加卷积层、激活函数、批量归一化和最大池化层
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

        # 添加全连接层、激活函数和Dropout
        self.layers.append(nn.Linear(8192 * 4, _denseSize1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nn.Linear(_denseSize1, _denseSize2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.5))

        last_denseSize = _denseSize2

        # 将卷积层和全连接层转换为ModuleList，方便管理
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.layers = nn.ModuleList(self.layers)

        # 添加两个全连接层，分别用于哈希和分类
        self.fc_plus = nn.Linear(_denseSize1, _hashSize)
        self.fc = nn.Linear(_hashSize, numCluster, bias=False)


    def forward(self, x):

        # 在输入数据的第二个维度增加一个维度，以匹配卷积层的期望输入
        x = x.unsqueeze(dim=1)
        # 依次通过卷积层
        for l in self.conv_layers:
            x = l(x)
        # 将卷积层的输出展平，以便输入到全连接层
        x = x.view(x.shape[0], -1)
#       for l in self.layers:
#            x = l(x)
        x = self.layers[0](x)
        # 将全连接层的输出通过哈希层
        x = self.fc_plus(x)
        # 使用自定义的Hash函数进行哈希
        code = GreedyHashLoss.Hash.apply(x)
        # 将哈希结果通过分类层
        output = self.fc(code)
        # 返回输出、全连接层的输出和哈希结果
        return output, x, code

# 输出加载模型的消息
print("Model Loading")

# 创建RevisedNetwork类的实例
model = RevisedNetwork()

# 从文件中加载预训练的模型权重
model.load_state_dict(torch.load(filename))

# 将模型设置为评估模式，关闭dropout等只在训练时使用的层
model.eval()

# 定义InferNetwork类，用于模型推理
class InferNetwork(torch.nn.Module):
    def __init__(self):
        super(InferNetwork, self).__init__()
        # 初始化卷积层列表，这里应该是具体的卷积层，但代码中未给出具体定义
        self.conv_layers = None  # 需要将其替换为实际的卷积层
        # 初始化全连接层列表，这里也应该是具体的全连接层，但代码中未给出具体定义
        self.layers = None  # 需要将其替换为实际的全连接层列表
        # 初始化额外的全连接层或操作，这里也未给出具体定义
        self.fc_plus = None  # 需要将其替换为实际的全连接层或操作

    def forward(self, x):
        # 增加数据的维度以适应卷积层的输入要求
        x = x.unsqueeze(dim=1)
        # 通过卷积层处理数据
        # 假设self.conv_layers是一个包含多个卷积层的列表
        for l in self.conv_layers:
            x = l(x)  # 对x应用卷积层l
        # 将卷积层的输出展平，以便输入到全连接层
        x = x.view(x.shape[0], -1)
        # 通过全连接层列表处理数据
        x = self.layer(x)
        # 通过额外的全连接层或操作处理数据
        x = self.fc_plus(x)
        # 获取x的符号函数结果，通常用于二值化哈希码
        x = x.sign()
        # 返回处理后的数据
        return x

# 输出信息，表示正在将模型保存到 PyTorch 格式
# Save InferNet
print("Model Saving to PT")
# 创建 InferNetwork 类的实例
infer = InferNetwork()
# 将修订模型的卷积层复制到推理模型中
infer.conv_layers = model.conv_layers
# 假设模型只有一个全连接层，将其复制到推理模型中
infer.layer = model.layers[0]
# 将修订模型的额外全连接层（或操作）复制到推理模型中
infer.fc_plus = model.fc_plus
# 将推理模型设置为评估模式，关闭如 dropout 等只在训练时使用的层
infer.eval()
# 使用 torch.jit.script 将推理模型转换为 TorchScript，这样可以优化模型性能并使其更容易部署
sm = torch.jit.script(infer)
# 保存 TorchScript 模型到指定的文件名（扩展名为 .pt）
sm.save("{}.pt".format(filename))

#sys.exit()

# Evaluation ############################################################################
# 遍历 infer.conv_layers 中的所有卷积层，并打印它们的信息。
for l in infer.conv_layers:
    print(l)
# 打印 infer.layer，这应该是模型的一个全连接层。
print(infer.layers)

# files_to_hash 函数定义开始
# 这个函数接受一个文件列表（files）和一个采样率（sampling）作为参数。
# 函数返回一个基于这些输入的SHA-256哈希值的前16个字符。
def files_to_hash(files, sampling):
    # 创建一个SHA-256哈希对象。
    m = hashlib.sha256()

    # 更新哈希对象，使用采样率的字符串编码。
    m.update(str(sampling).encode())

    # 对文件列表进行排序，并逐个更新哈希对象，使用文件名的字符串编码。
    for fn in sorted(files):
        m.update(fn.encode())

    # 返回哈希值的十六进制表示的前16个字符。
    return m.hexdigest()[:16] # Use the first 16 chars. It's too long 使用前16个字符。原始哈希值太长


# Use it when the data is small enough to store in the main memory
# 定义一个名为Loader的类，用于加载数据
class Loader:
    # 初始化方法，当创建Loader类的实例时会被调用
    def __init__(self, files, sampling=1.0):
        # 根据文件列表和采样率生成一个唯一的picklename，用于后续存储预处理后的数据
        picklename = "preprocessed_data/" + files_to_hash(files, sampling) + ".data"

        # 检查pickle文件是否已存在
        if os.path.exists(picklename):
            # 如果文件存在，则加载已预处理的数据
            with open(picklename, 'rb') as f:
                # 使用pickle模块的load方法从文件中加载数据
                self.alldata = pickle.load(f)
                # 从加载的数据中获取数据的长度（可能是样本数量或其他度量）
                self.alllen = self.alldata[2]
        else:
            # 如果文件不存在，则进行数据处理
            # 复制文件列表到dataset，以便后续可以对其进行修改
            self.dataset = files.copy()
            # 设置随机种子，以确保每次随机打乱数据集时结果一致
            random.seed(1)
            # 随机打乱数据集
            random.shuffle(self.dataset)
            # 如果指定了采样率且小于1.0，则仅保留数据集的一部分
            if sampling < 1.0:
                self.dataset = self.dataset[:int(len(self.dataset)*sampling)]
                # 加载数据，这里假设load_data方法是从其他来源（如磁盘）加载数据
            self.alldata = self.load_data()
            # 获取数据的长度（可能是样本数量或其他度量）
            self.alllen = self.alldata[2]
            # 将处理后的数据保存到pickle文件中，以便下次可以直接加载而无需重新处理
            with open(picklename, 'wb') as f:
                pickle.dump(self.alldata, f)

        # 初始化一个计数器，可能用于迭代访问数据
        self.num = 0

# 加载数据的实现方法
def load_data(self):
    # 获取数据集的长度
    alllen = len(self.dataset)
    # 初始化一个列表，用于存储每个数据文件的处理结果
    ret1 = [[] for i in range(alllen)]
    # 初始化一个空列表，用于存储从文件名中提取的某种标识符
    ret2 = []

    # 遍历数据集
    for i in range(alllen):
        # 以二进制模式打开数据文件
        with open(self.dataset[i], 'rb') as f:
            # 读取文件内容
            data = f.read()
            # 将文件内容从字节转换为整数列表
            data = [int(d) for d in data]
            # 将处理后的数据存储在ret1的对应位置
            ret1[i] = data
            # 从文件名中提取标识符（这里假设文件名格式为“.../some_number_...”，并提取some_number）
            fn = self.dataset[i]
            ret2.append(int(fn[fn.rfind('/') + 1:fn.rfind('_')]))
    # 将ret1转换为torch张量，并进行归一化处理（减去128后除以128.0）
    # 这里假设数据是图像数据，且原始像素值在0-255之间，这样处理可以将它们归一化到-1到1之间
    return ((torch.tensor(ret1) - 128) / 128.0), torch.tensor(ret2), alllen



# 使得Loader类的实例可以迭代
def __iter__(self):
    # 返回Loader类的实例本身，以便在for循环中使用
    return self

# 定义迭代器的下一个元素获取方法
def __next__(self):
    # 获取当前的索引
    num = self.num
    # 计算当前批次的大小，不超过剩余数据量和预设的BATCH_SIZE
    nowlen = min(self.alllen - num, BATCH_SIZE)
    # 如果还有剩余数据可以处理
    if nowlen > 0:
        # 更新当前索引
        self.num += nowlen
        # 从alldata中获取对应的数据和标识符
        ret1 = self.alldata[0][num:num+nowlen]
        ret2 = self.alldata[1][num:num+nowlen]
        # 将获取的数据和标识符打包成返回结果
        ret = [ret1, ret2]
        # 如果使用了CUDA，将数据转移到GPU上
        for i in range(len(ret)):
            ret[i] = ret[i].cpu()
            # 返回当前批次的数据和标识符

        return ret

    else:

        # 如果没有剩余数据，则抛出StopIteration异常，结束迭代

        raise StopIteration

'''

train_data = []
test_data = []

random.seed(1)
for i in range(numCluster):
    name = glob(os.path.join(path,'{}/*'.format(i)))

    random.shuffle(name)
    n = len(name)
    
    train_data += name[:n // 10]
    test_data += name[n // 10:]

print("Train data:", len(train_data))
print("Test data:", len(test_data))

# Evaluation 1: Training data validation
train_data_tensor = list(Loader(train_data))
infer = infer.cuda()

print("Generating Hash Dict")
hashdict = defaultdict(list)
with torch.no_grad():
    for data, target in tqdm(train_data_tensor):
        out = infer.forward(data)
        tvecs = out.cpu().detach().numpy()
        for t, l in zip(tvecs, target):
            hval = ''.join(['1' if x>=0.0 else '0' for x in t])
            hashdict[int(l)].append(hval)

def compute_score(hashdict):
    score = 0
    score_dict = dict()
    for k in hashdict.keys():
        n_hash = len(hashdict[k])
        max_score_per_cluster = 0
        for i in range(n_hash):
            cur_score = 0
            for j in range(n_hash):
                if i == j:
                    cur_score += 1
                    continue

                if hashdict[k][i] == hashdict[k][j]:
                    cur_score += 1

            if max_score_per_cluster < cur_score:
                max_score_per_cluster = cur_score

        score_dict[k] = max_score_per_cluster
        score += max_score_per_cluster

    return score, score_dict

score = None
score, score_dict = compute_score(hashdict)

with open(hashdict_filename, 'w') as f:
    total_size = 0
    for k in hashdict.keys():
        n_hash =  len(hashdict[k])
        total_size += n_hash

        f.write("Class\t{}\t{}\t{}\n".format(k, score_dict[k], n_hash))
        for hval in hashdict[k]:
            f.write("{}\n".format(hval))


    if score is not None:
        f.write("Total Score: {} / {}\n".format(score, total_size))
    

# Evalution 2: BLK data validation
sys.exit()
def read_blkfile_to_tensor(filename, n):
    ret = []
    with open(filename, 'rb') as f:
        for i in range(n):
            data = f.read(4096)
            data = [int(d) for d in data]
#print(', '.join(map(lambda x: str(x), data)))
            ret.append(data)
    return ((torch.tensor(ret)-128)/128.0)
 
N = 100
inp = read_blkfile_to_tensor("/home/compu/jeonggyun/ts/build/mix10", N)
#print(inp[0])
with torch.no_grad():
    out = infer.forward(inp.cuda())

for i in range(N):
#print(', '.join(map(lambda x: "%.2f" % x, inp[i].cpu().detach().numpy()[:10])))
    print(', '.join(map(lambda x: "%.6f" % x, out[i].cpu().detach().numpy()[:10])))
    t = [1 if x >=0.0 else 0 for x in t]
    print(''.join(map(lambda x: str(x), t)))
'''
