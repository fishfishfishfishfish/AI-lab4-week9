import numpy
import math


def get_best_attribute(data):  # 输入numpy.array
    best_attribute = 0
    best_attribute_value = 0
    row, col = data.shape
    for i in range(col-1):  # 最后一列是分类结果
        temp_value = cal_id3(data[:, col-1], data[:, i])
        if temp_value > best_attribute_value:
            best_attribute = i
            best_attribute_value = temp_value
    return best_attribute


def cal_id3(a_col, r_col):
    a_col = list(a_col.copy())
    r_col = list(r_col.copy())
    origin_entropy = entropy(r_col)
    col_size = len(r_col)
    labels = set(a_col)
    res = 0
    for label in labels:
        prop = a_col.count(label)/col_size
        sub_col = []
        for i in range(col_size):
            if a_col[i] == label:
                sub_col.append(r_col[i])
        condition_entropy = entropy(sub_col)
        res += prop*condition_entropy
    return origin_entropy - res


def entropy(tag_col):  # 输入数组
    tag_col = list(tag_col)
    col_size = len(tag_col)
    value_type = set(tag_col)
    res = 0
    for t in value_type:
        cnt = tag_col.count(t)
        res -= (cnt/col_size)*math.log(cnt/col_size, 2)
    return res


print('使用ID3，训练集请命名为‘train.csv’')
f = open('train.csv', 'r')
data = []
for line in f.readlines():
    row = []
    t_row = line.split(',')
    for t in t_row:
        row.append(int(t))
    data.append(row)
data = numpy.array(data)
print('选择第', get_best_attribute(data), '个属性')
