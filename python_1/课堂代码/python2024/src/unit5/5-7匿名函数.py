#5-7匿名函数

# function1=lambda x:x*2
# result1=function1(6)
# print(result1)

# function2=lambda a,b:a+b
# result2=function2(6,3)
# print(result2)

#将函数作为参数
listA=[1,2,3,4,5]
newListA=filter(lambda x:x%2==0,listA) #筛选列表数据
print(list(newListA))