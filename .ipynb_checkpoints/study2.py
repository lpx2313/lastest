import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,								# 这里表示加载训练集
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,							# 这里表示不加载训练集
    download=True,
    transform=ToTensor()
)

# 定义一个标签映射字典labels_map
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# 创建一个名为figure的图像对象，设置大小为8*8英寸
figure = plt.figure(figsize=(8, 8))
# 定义要显示的子图列数和行数(如下图所示)
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    # 此处，len(training_data)用于返回训练集长度，size = (1,).item指定生成一个一维
    # 随机数张量并将其转换为python标量用于获取索引值（int型）
    sample_idx = torch.randint(len(training_data), size=(1,)).item()

    # 在pytorch中，使用索引访问一个数据集对象时，该对象返回的是一个tuple包含图像+标签
    img, label = training_data[sample_idx]

    # 添加子图到figure中
    figure.add_subplot(rows, cols, i)

    # 设置子图标题为label对应标签
    plt.title(labels_map[label])
    plt.axis("off")  # 关闭子图的坐标轴显示，使子图不显示坐标轴刻度和标签

    # 将压缩后的图像数据以灰度映射的形式显示
    plt.imshow(img.squeeze(), cmap="gray")  # squeeze用于压缩图像维度
plt.show()

from torch.utils.data import DataLoader

# # 这两个玩意就像一个迭代器，按照你的规矩一次给你多少东西
# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#
# # 展示从DataLoader中加载的一个样本的图像和标签
# # next(iter(...))即从迭代器中获取下一个元素，即获取DataLoader中的一个批次数据
# train_features, train_labels = next(iter(train_dataloader))
#
# # 获取图像和标签的维度信息，以方便知道如何进行处理
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
#
# # 使用squeeze()函数将图像维度去除以满足imshow()函数的输入要求
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")



import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#  这是新分支
# MNIST数据集中默认图像格式是PIL，标签格式是整数，此处我们编写代码将图像转换为tensor，
# 标签转化为独热编码（one-hot encoded）
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    # 此处定义了一个lambda函数，将标签转换为一个大小为10的全0张量
    # torch.zeros(x, dtype = y)创建一个大小为x类型为y的全零张量
    # 使用scatter_(x, y, value = z)函数将位置y的元素更改为数值z，x表示在哪个维度更改，这里为行
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float)
    .scatter_(0, torch.tensor(y), value=1))
)

x: np.ndarray = np.array([1, 2, 3]).reshape([3, 1])
print(x.shape)
import numpy as np

print(np.char.replace(['i like runoob', "lpx sdfsa sss da"], ['oo', 's'], 'cc'))

import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1, 11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x, y)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
# 计算正弦曲线上点的 x 和 y 坐标
x = np.arange(0,  3  * np.pi,  0.1) 
y = np.sin(x)
plt.title("sine wave form")
# 使用 matplotlib 来绘制点
plt.plot(x, y)
plt.show()