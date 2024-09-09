import pandas as pd

# mydataset = {
#   'sites': ["Google", "Runoob", "Wiki"],
#   'number': [1, 2, 3]
# }
#
# myvar = pd.DataFrame(mydataset)
#
# print(myvar)
#
# a = [1, 2, 3]
#
# myvar = pd.Series(a)
#
# print(myvar)
# print(myvar.loc[0])
#
# # 指定索引创建 Series
# s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
#
# # 获取值
# value = s[2]  # 获取索引为2的值
# print(s['a'])  # 返回索引标签 'a' 对应的元素
#
# # 获取多个值
# subset = s[1:4]  # 获取索引为1到3的值
#
# # 使用自定义索引
# value = s['b']  # 获取索引为'b'的值
#
# # 索引和值的对应关系
# for index, value in s.items():
#     print(f"Index: {index}, Value: {value}")
#
#
# # 使用切片语法来访问 Series 的一部分
# print(s['a':'c'])  # 返回索引标签 'a' 到 'c' 之间的元素
# print(s[:3])  # 返回前三个元素
#
# # 为特定的索引标签赋值
# s['a'] = 10  # 将索引标签 'a' 对应的元素修改为 10
#
# # 通过赋值给新的索引标签来添加元素
# s['e'] = 5  # 在 Series 中添加一个新的元素，索引标签为 'e'
#
# # 使用 del 删除指定索引标签的元素。
# del s['a']  # 删除索引标签 'a' 对应的元素
#
# # 使用 drop 方法删除一个或多个索引标签，并返回一个新的 Series。
# s_dropped = s.drop(['b'])  # 返回一个删除了索引标签 'b' 的新 Series
# import pandas as pd
#
# data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]
#
# # 创建DataFrame
# df = pd.DataFrame(data, columns=['Site', 'Age'])
#
# # 使用astype方法设置每列的数据类型
# df['Site'] = df['Site'].astype(str)
# df['Age'] = df['Age'].astype(float)
#
# print(df)
#
#
# import pandas as pd
#
# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }
#
# # 数据载入到 DataFrame 对象
# df = pd.DataFrame(data)
#
# # 返回第一行
# import pandas as pd
#
# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }
#
# df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
#
# # 指定索引
# print(df.loc["day2"]["duration"])
#
# print("----------------------")
# import numpy as np
#
# # 通过 NumPy 数组创建 DataFrame
# df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index=["r1", "r2", "r3"], columns=["c1", "c2", "c3"])
# print(df)
# print(df.mean())    # 求平均值
# print(df.sum())

import pandas as pd

df = pd.read_csv('property-data.csv')
#
# print (df['NUM_BEDROOMS'])
# print (df['NUM_BEDROOMS'].isnull())

missing_values = ["n/a", "na", "--"]
df = pd.read_csv('property-data.csv', na_values = missing_values)

# print (df['NUM_BEDROOMS'])
# print (df['NUM_BEDROOMS'].isnull())

import pandas as pd

df = pd.read_csv('property-data.csv')
print(df)
print("===================")
new_df = df.dropna()

print(new_df.to_string())