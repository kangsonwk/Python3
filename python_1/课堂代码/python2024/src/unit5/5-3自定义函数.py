#5-3自定义函数

# #定义函数
# def function1():
#     print("函数function1被执行了！")
#     print("---执行完毕！")
#
# def function2():
#     print("在function2中调用function1:")
#     function1()

#调用函数
# function1()
# for i in range(100):
#     function1()

# function2() #在函数中调用函数


#-----------------------------------
def getSum(a,b):
    result=a+b
    print("相加结果为：",result)
    return result #返回值

getSum(4,5)
r=getSum(3,12)
print(r/2)





