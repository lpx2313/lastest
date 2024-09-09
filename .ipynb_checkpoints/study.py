import torch
import numpy as np


if __name__ == '__main__':
    # 方式1：直接使用数据初始化
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)

    # 方式2：使用Numpy数组
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    # 方式3：使用另一个tensor
    x_ones = torch.ones_like(x_data)
    print(f"Ones Tensor: \n {x_ones} \n")  # 保留了x_data维度、数据类型的任意1值张量
    x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆写了x_data数据类型为float，维度不变的任意值张量
    print(f"Random Tensor: \n {x_rand} \n")

    # shape定义了张良的维度
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")  # 生成元素为随机值的符合shape维度的张量
    print(f"Ones Tensor: \n {ones_tensor} \n")  # 生成元素为1的符合shape维度的张量
    print(f"Zeros Tensor: \n {zeros_tensor}")  # 生成元素为0的符合shape维度的张量

    # 创建一个任意的(3, 4)维度的张量，展示其属性如下
    tensor = torch.rand(3, 4)
    tensor = tensor.to('cuda')
    print(f"Shape of tensor: {tensor.shape}")  # 张量的维度
    print(f"Datatype of tensor: {tensor.dtype}")  # 张量的数据类型
    print(f"Device tensor is stored on: {tensor.device}")  # 张量的存储设备

    # tensor的操作大多类似于numpy：
    tensor = torch.ones(4, 4)
    print('First row: ',tensor[0])
    print('First column: ', tensor[:, 0])
    print('Last column:', tensor[..., -1])
    tensor[:,1] = 0
    print(tensor)

    # 输出为
    # First row:  tensor([1., 1., 1., 1.])
    # First column:  tensor([1., 1., 1., 1.])
    # Last column: tensor([1., 1., 1., 1.])
    # tensor([[1., 0., 1., 1.],
    #         [1., 0., 1., 1.],
    #         [1., 0., 1., 1.],
    #         [1., 0., 1., 1.]])

    t1 = torch.cat([tensor, tensor, tensor], dim=0)
    print(t1)


    # 两个张量之间的矩阵乘法，即tensor与其转置之间的乘积
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    print(y1)
    print(y2)
    y3 = torch.rand_like(tensor)



    # 两个张量对应位置元素相乘，类似于点乘，如果累加起来就是内积
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
    print(z1)
    print(z2)
    z3 = torch.rand_like(tensor)


    agg: torch.Tensor= tensor.sum()
    print(agg, type(agg))  # 为什么求和之后还是tensor标量，因为到目前为止还是可以计算梯度，所以不可直接变基本数据类型
    agg.backward()
    agg_item = agg.item()
    print(agg_item, type(agg_item))  # 输出：12.0 <class 'float'>

    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()   # 可以直接转numpy
    print(f"n: {n}")

    # tensor与numpy共用一个底层内存，此处更改tensor对应n的值也会修改
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")
    # 由此可见，tensor底层存储数据用的是ndarray，只是在此之上增加了很多功能，理解为np数组是tensor对象的一个属性
    # class Tensor：
    #   data = np.array(xxx)
    #   其它功能，比如梯度

    n = np.ones(5)
    t = torch.from_numpy(n)

    # tensor与numpy共用一个底层内存，此处更改n对应t的值也会修改
    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")

    x = torch.tensor(3, dtype=torch.float32, requires_grad=True)
    y1 = x ** 2
    y2 = 2 * x + 5
    y3= y2.detach()
    z: torch.Tensor = y1 + 2 * y3
    # z.backward()
    # y3.backward()
    print(x.grad)












