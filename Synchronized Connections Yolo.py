from common import *


class MyModel(nn.Module):
    def __init__(self, inchannel=3, classnum=10):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            RepConvN(inchannel, 64, 3, 2),
            ResBlock(64, 256, 3, 2, 2, 1, False),
            ResBlock(256, 512, 3, 2, 1, 1, False),
            ResBlock(512, 640, 3, 2, 1, 1, False),
            ResBlock(640, 2048, 3, 2, 1, 1, True),
            # 平均值汇聚层
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(2048, classnum)
        )
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv1d):
                # kaiming正态分布初始化，使得Conv2d卷积层反向传播的输出的方差都为1
                # fan_in：权重是通过线性层（卷积或全连接）隐性确定
                # fan_out：通过创建随机矩阵显式创建权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inx):
        iny = self.net(inx)
        return iny
