# （二）使用torch完成一个标准的神经网络（共45分）
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 1.导入torch，设置随机种子
torch.manual_seed(1)
# 2.正确定义出使用的X和Y数据集：
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_train = np.array([[0], [1], [1], [0]], dtype=np.float32)
# 3.根据给定的X，Y，定义X，Y变量
x=torch.from_numpy(x_train)
y=torch.from_numpy(y_train)
# 4.使用torch 构建2个linear
# 5.使用torch构造sigmoid
# 6.将模型的这几层堆叠到一起
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.linear1 = nn.Linear(2, 10)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(10, 1)
    def forward(self,x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x
model=mynet()
# model=nn.Sequential(
#     nn.Linear(2,10),
#     nn.Sigmoid(),
#     nn.Linear(10,1),
#     nn.Sigmoid()
# )
cost=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.02)
# 7.总共训练1000次
# 8.定义出代价函数，并计算代价函数的值
# 9.传入模型训练
# 10.并且更新参数，并且将梯度清零
for epoch in range(1000):
    h=model(x)
    loss=cost(h,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# 11.间隔100打印并输出loss
    if (epoch+1) % 100 ==0:
        print('loss:{:.4f}'.format(loss.item()))


