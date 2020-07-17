import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#
# pytorch网站教学案例
#


# =======Define the network=======
class Net(nn.Module):
    # ——————————————————————————————————————————
    # 定义Net的初始化函数，定义该神经网络结构，计算图
    # ——————————————————————————————————————————
    def __init__(self):
        # 初始化父类，先运行nn.Module的初始化函数
        super(Net, self).__init__()

        # 两层图像卷积层
        # 第一层（输入：1通道图像（灰度图），输出：6通道图像， 卷积3x3）
        # 第二层（输入：6通道图像，输出：16通道图像， 卷积3x3）
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # 三层全连接层，连接函数为线性函数
        # 第一层（输入： 16x6x6 个节点， 输出： 120 个节点）
        # 第二层（输入： 120 个节点， 输出： 84 个节点）
        # 第三层（输入： 84 个节点， 输出： 10 个节点）
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # ——————————————————————————————————————————————————————————
    # 定义该神经网络的向前传播函数，向后传播函数也会自动生成（autograd）
    # ——————————————————————————————————————————————————————————
    def forward(self, x):
        # 前向传播

        # 两步都是先卷积然后池化（案例里用一行写了两步），ReLU为激活函数
        # 第一步：输入x经过卷积conv1之后，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        # 第二步：输入x经过卷积conv2之后，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = x.view(-1, self.num_flat_features(x))

        # 以下三步都是激活全连接层，将x通过定义的全连接层不断向后传递
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # ——————————————————————————————————————————————————————————
    # 计算x特征总量
    # ——————————————————————————————————————————————————————————
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# 此处打印的是网络结构信息
print(net)

# 此处查看的是网络中w，b等各参数的信息
params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
net.zero_grad()
out.backward(torch.randn(1, 10))

# =======Loss Function=======
print("**********out************")
output = net(input)
print(output)
# 设定一个虚拟的预期结果
target = torch.randn(10)
target = target.view(1, -1)
print("**********target**********")
print(target)
# 设置损失函数为均方误差
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# =======Backdrop=======
# 将所有参数的梯度缓存区清零
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# =======Update the weights=======

# 设定随机梯度下降方法，学习率0.01
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 训练过程:
# 未优化前损失函数值
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
print(loss)
# 优化一步
loss.backward()
optimizer.step()
# 优化一步后损失函数值
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
print(loss)

# 使用循环几步优化查看loss函数值变化
for i in range(0, 10):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    optimizer.step()
    i += 1
