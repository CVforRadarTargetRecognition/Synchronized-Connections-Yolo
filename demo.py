from RepGelan import MyModel
from common import *
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


train_loader = torch.load('train_loader_snr30.pt')
test_loader = torch.load('test_loader_snr30.pt')

# 创建子类的实例，并搬到GPU 上
inchannels = 3
classnums = 15
ls = True
model = MyModel(inchannel=inchannels, classnum=classnums).to('cuda:0')


def test(in_model, data_loader):
    # 测试网络
    correct_num = 0
    total_num = 0
    in_model.eval()
    with torch.no_grad():  # 该局部关闭梯度计算功能
        for (inx, iny) in data_loader:  # 获取小批次的x 与y
            inx, iny = inx.to('cuda:0'), iny.to('cuda:0')
            result = in_model(inx)  # 一次前向传播（小批量）
            _, predicted = torch.max(result.data, dim=1)
            correct_num += torch.sum((predicted == iny.reshape(-1)))
            total_num += iny.size(0)
    return 100*correct_num/total_num


# 损失函数的选择
if not ls:
    loss_fn = nn.CrossEntropyLoss()  # 自带softmax 激活函数
else:
    loss_fn = LabelSmoothingLoss(classes=classnums, smoothing=0.005)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=1e-2,
                            momentum=0.9,
                            weight_decay=1e-4
                            )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    T_max=250)

# 训练网络
start_epoch = -1
epochs = 250
times = []
losses = []  # 记录损失函数变化的列表
accuracies = []
MaxAcc = torch.tensor([0]).to('cuda:0')
lr_es = []

RESUME = False
if RESUME:
    checkpoint = torch.load('checkpoint.pth')  # 加载断点

    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    scheduler.load_state_dict(checkpoint['scheduler'])
    accuracies = torch.load("accuracies.pt")
    times = torch.load("times.pt")
    lr_es = torch.load("lr_es.pt")
    MaxAcc, _ = torch.max(torch.tensor(accuracies).reshape(-1, 1), dim=0)
    MaxAcc = MaxAcc.to('cuda:0')

for epoch in range(start_epoch+1,epochs):
    T1 = time.perf_counter()
    model.train()
    print(epoch)
    for (x, y) in train_loader:  # 获取小批次的x 与y
        x, y = x.to('cuda:0'), y.to('cuda:0')
        Pred = model(x)  # 一次前向传播（小批量）
        if ls: # 使用标签平滑
            loss = loss_fn(Pred, y)  # 计算损失函数
        else: # 不使用标签平滑
            with torch.no_grad():
                true_dist = torch.zeros_like(Pred)  # 创建与预测相同形状的张量
                # 在真实类别的位置上设置置信度
                true_dist.scatter_(1, y.data.reshape(-1).unsqueeze(1), 1)
            loss = loss_fn(Pred, true_dist)  # 计算损失函数
        losses.append(loss.item())  # 记录损失函数的变化
        optimizer.zero_grad()  # 清理上一轮滞留的梯度
        loss.backward()  # 一次反向传播
        optimizer.step()  # 优化内部参数
        # scheduler.step()
    T2 = time.perf_counter()
    times.append(T2 - T1)
    # print('程序运行时间:%s秒' % (T2 - T1))
    accuracy = test(model, test_loader)
    accuracies.append(accuracy)
    scheduler.step()
    lr_es.append(scheduler.get_last_lr())
    print(f'当前学习率: {scheduler.get_last_lr()}')
    print(f'训练集精准度: {accuracy} %')
    # print(losses[-1])
    torch.save(times, "times.pt")
    torch.save(accuracies, "accuracies.pt")
    torch.save(lr_es, "lr_es.pt")
    if accuracy > MaxAcc:
        MaxAcc = accuracy
        torch.save(model, 'MaxAccRes.ph')  # 保存网络
    if epoch >= 245:
        torch.save(model, '%s.pth' % (str(epoch)))  # 保存网络
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, 'checkpoint.pth')
