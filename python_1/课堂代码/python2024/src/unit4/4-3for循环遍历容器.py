# 4-3 for循环遍历容器

names=["张三","李四","王五","赵六"]

# #直接遍历
# for name in names:
#     print(name)

# #构造索引
# for i in range(0,len(names)):
#     print(names[i])


#遍历元组
scores=(67,78,68,87,92,45,69,77)

# for i in range(0,len(scores)):
#     print(scores[i])

# #平均分
# total=0
# for score in scores:
#     total=total+score
# print(total/len(scores))

# #遍历字典 只获取键
# dictA={"name":"张三","age":18,"hobby":"打球"}
# for x in dictA:
#     print(x,dictA[x])

#遍历集合
setA={1,324,243,535,23}
for i in setA:
    print(i)

