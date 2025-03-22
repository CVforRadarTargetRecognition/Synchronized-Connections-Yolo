import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import scipy.io as sio
import numpy as np
import math


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        """
        初始化标签平滑损失函数。
        :param classes: 类别的总数。
        :param smoothing: 标签平滑的系数，默认为0.1。
        :param dim: 计算softmax的维度，默认为最后一个维度。
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing  # 置信度，即非平滑部分的权重
        self.smoothing = smoothing  # 平滑系数
        self.cls = classes  # 类别总数
        self.dim = dim  # softmax操作的维度

    def forward(self, pred, target):
        """
        计算标签平滑损失。
        :param pred: 模型的预测输出，未经softmax处理。
        :param target: 真实的标签。
        :return: 平滑损失的平均值。
        """
        # 对预测结果应用log softmax
        pred = pred.log_softmax(dim=self.dim)
        # 生成平滑的目标分布
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)  # 创建与预测相同形状的张量
            # 对所有类别填充平滑值
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # 在真实类别的位置上设置置信度
            true_dist.scatter_(1, target.data.reshape(-1).unsqueeze(1), self.confidence)
        # 计算平滑损失
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // ratio),
                                nn.SiLU(),
                                nn.Linear(in_planes // ratio, in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MyData(Dataset):  # 继承Dataset 类
    def __init__(self, filepath, k, mode, channels, dimens=2):
        datafile = sio.loadmat(filepath)  # 导入数据
        temp_data = datafile['tempdata']  # 提取数据集
        temp_dataset = temp_data  # 间隔0.25°的样本
        # 训练集测试集各50%
        if mode:  # 训练集
            temp_dataset = temp_dataset[::2, :]
        else:
            temp_dataset = temp_dataset[1::2, :]
        x, y = temp_dataset.shape
        y_new = int((y - 1) / 5)
        origin_dataset = np.zeros((x, channels, y_new))  # 三个通道依次为HRRP、频谱和功率谱
        origin_label = np.zeros((x, 1))
        tempcnt = 0
        for i in range(x):
            tempcnt = tempcnt + 1
            origin_label[i, 0] = temp_dataset[i, -1]
            for j in range(y_new):
                origin_dataset[i, 0, j] = (temp_dataset[i, j] ** 2 + temp_dataset[i, j + 128] ** 2) ** 0.5
                if channels > 1:
                    origin_dataset[i, 1, j] = (temp_dataset[i, j + 256] ** 2 + temp_dataset[i, j + 384] ** 2) ** 0.5
                    if channels > 2:
                        origin_dataset[i, 2, j] = temp_dataset[i, j + 512]
        datarray = origin_dataset[:tempcnt].astype(np.float32)  # 转为float32类型数组
        data_tensor = torch.tensor(datarray)  # 数组转为张量
        label_array = origin_label[:tempcnt].astype(np.int64)
        label_tensor = torch.tensor(label_array)  # 数组转为张量
        self.len = tempcnt  # 样本的总数
        self.Y = label_tensor
        if dimens <= 1:
            self.X = data_tensor

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


def GenerateDataSet(filename, k, channels, dimens=2):  # 文件读取路径、缩放倍数、通道数
    # 划分训练集与测试集
    train_data = MyData(filepath=filename, k=k, mode=1, channels=channels, dimens=dimens)  # 生成训练集
    test_data = MyData(filepath=filename, k=k, mode=0, channels=channels, dimens=dimens)  # 生成测试集
    train_size = int(len(train_data))  # 训练集的样本数量
    test_size = train_size  # 测试集的样本数量
    temp_dataset = torch.cat((train_data.X, test_data.X), 0)
    for i in range(channels):  # 遍历数据集每个通道，分别进行标准化
        mean_value = torch.mean(temp_dataset[:, i, :])  # 训练集的均值
        std_value = torch.std(temp_dataset[:, i, :])  # 训练集的标准差
        train_data.X[:, i, :] = (train_data.X[:, i, :] - mean_value) / std_value  # 数据集标准化
        test_data.X[:, i, :] = (test_data.X[:, i, :] - mean_value) / std_value  # 数据集标准化
    train_data, _ = random_split(train_data, [train_size, 0])
    _, test_data = random_split(test_data, [0, test_size])
    return train_data, test_data


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvO(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x):
        return self.conv(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool1d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False, LastLayer=False):
        super().__init__()
        assert k == 3 and p == 1
        self.lastlayer = LastLayer
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm1d(c2)
        self.bn_flag = bn and c1 == c2
        self.conv1 = ConvO(c1, c2, k, s, p=p, g=g)
        self.conv2 = ConvO(c1, c2, 1, s, p=(p - k // 2), g=g)

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn_flag is False else x
        y = self.conv1(x) + self.conv2(x) + id_out
        return y if self.lastlayer else self.act(self.bn(y))


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1, bn=True)
        self.cv2 = RepConvN(c_, c2, k[1], 1, g=g, bn=True, LastLayer=True)
        self.add = shortcut and c1 == c2
        self.bn = nn.BatchNorm1d(c2)
        self.cv3 = ConvO(c1, c2, 1, 1)

    def forward(self, x):
        y = x + self.cv2(self.cv1(x)) + self.cv3(x) if self.add else self.cv2(self.cv1(x)) + self.cv3(x)
        return F.silu(self.bn(y))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = ConvO(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.cv4 = ConvO(c1, c2, 1, 1)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.add = True if c1 == c2 else False
        self.bn = nn.BatchNorm1d(c2)

    def forward(self, x):
        y = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        return F.silu(self.bn(y + x + self.cv4(x))) if self.add is True else F.silu(self.bn(y + self.cv4(x)))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1, a=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv4 = ConvO(c3+(2*c4), c2, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), RepConvN(c4, c4, 3, 1, bn=True))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), RepConvN(c4, c4, 3, 1, bn=True))
        self.bn = nn.BatchNorm1d(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ResBlock(nn.Module):
    def __init__(self, c1, c2, k, s, e, n, a=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c1 * e)
        self.s = s
        self.a = a
        self.conv1 = RepConvN(c1, self.c, k, s) if k == 3 else Conv(c1, self.c, k, s)
        self.conv2 = ConvO(c1, c2, 1, s)
        self.conv3 = ConvO(self.c, c2, 1, 1)
        self.RepNCSPELAN = RepNCSPELAN4(self.c, c2, self.c, self.c // 2, c5=n, a=a)
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention(kernel_size=7)
        self.bn = nn.BatchNorm1d(c2)

    def forward(self, x):
        y1 = self.conv1(x)
        y = self.RepNCSPELAN(y1)
        if self.a:
            y2 = y + self.conv2(x) + self.conv3(y1) if self.s == 2 else y + x + self.conv3(y1)
            y = self.ca(y2) * y2
            y = self.sa(y) * y + y2
            return F.silu(y)
        else:
            y = y + self.conv2(x) + self.conv3(y1) if self.s == 2 else y + x + self.conv3(y1)
            return F.silu(self.bn(y))