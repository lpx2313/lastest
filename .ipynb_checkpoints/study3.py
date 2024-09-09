import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")


    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()  # 这已经是规范了，自定义的神经网络一定需要用到接口规范的所有东西，所以一定要用父类初始化函数
            # nn.Flatten()用于将二维图像数据展平为一维，以方便使用其作为全连接层的输入
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(  # 这里表示序列化执行,前一个的输出交给后一个输入，最后一个返回，这里面所有的类都重写了__call__魔术方法，并且里面forward
                nn.Linear(in_features=28*28, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


    model = NeuralNetwork().to(device)
    print(model)
    lr = 0.01
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    # requires_grad=True表示希望对该张量进行梯度计算，因为x，y是作为数值的，z是w，b的函数
    w = torch.randn(5, 3, requires_grad=True)	# 权重张量
    b = torch.randn(3, requires_grad=True)		# 偏置张量

    #  这是新分支
    # # 使用二进制交叉熵损失函数计算预测张量z和目标输出张量y之间的损失。
    # for i in range(5):
    #     z = torch.matmul(x, w) + b
    #     loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    #     loss_item = loss.item()
    #     print(f"第i轮loss： {loss_item}")
    #     loss.backward()
    #     w = w - w.grad * lr
    #     b = b - b.grad * lr
    #     # print(w.grad)  # 然后w里面每一个权值都可以反向更新
    #     # print(b.grad)

    z = torch.matmul(x, w) + b
    print(z.requires_grad)

    with torch.no_grad():  # 上下文管理器，临时禁用梯度跟踪
        z = torch.matmul(x, w) + b
    print(z.requires_grad)

    z = torch.matmul(x, w) + b
    z_det = z.detach()
    print(z_det.requires_grad)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵，这里只是举例，也一定实现了call

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 自动识别损失函数
    # 这里model.parameters()是什么，那么执行zero_grad()的时候就会清理对应的梯度，因为梯度会累积！！！





