#5-4参数和返回值

# def getSum(a,b): #形参
#     result=a+b
#     print("相加结果为：",result)
#     return result #返回值
#
# getSum(4,5) #实参
# #实参的个数必须和形参一致
#
# #返回值
# r=getSum(4,5)
# print(getSum(6,7))


# #返回多个值  以元组形式返回
# def function1():
#     print("function1被执行了")
#     a=2
#     b=3
#     c=4
#     return a,b,c
#
# result=function1()
# print(result)

#单独使用return-终止整个方法
def function2(a):
    print("function2被执行了")
    if a>10:
        print("大于10")
        return
    else:
        print("小于10")
    print("函数执行完毕！")
function2(8)
