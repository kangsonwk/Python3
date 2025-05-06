# 2-15集合
# 特性：无序；不重复；可以是不同数据类型；
seta = {321, 3, 42, 4, 3, 355, 35, 46, "hello"}
# print(seta)

# 列表去重
lista = [121, 2, 4, 33, 54646, 7, 575, 7, 76]
setb = set(lista)  # 将其他序列转换成set
listb = list(setb)  # 将其他序列转换成list
# print(listb)

# 集合运算
seta = {1, 2, 3, 4, 5, 6}
setb = {7, 8, 9, 4, 5, 6}
print(seta & setb)  # 交集
print(seta | setb)  # 并集
print(seta - setb)  # 差集
print(seta ^ setb)  # 对称差集
