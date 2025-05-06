# 2-16容器中的常用函数
listA=[6,21,34,253,53,645,757,57,58,68]
tupleA=(6,21,34,253,53,645,757,57,58,68,58,58)
dictA={"name":"张三","age":18,"hobby":"打球"}
setA={321,73,42,4,3,355,35,46}

#获取长度
print(len(setA))

#最大值，最小值，求和
print(max(listA))
print(min(listA))
print(sum(listA))

#列表排序
listA.sort()
print(listA)

listA.sort(reverse=True)
print(listA)

#统计元素出现的次数 --列表 元组
print(tupleA.count(58))

#通过值获取索引 --列表 元组
print(tupleA.index(58))

#字典获取值
print(dictA.get("name"))
print(dictA.get("sex"))














