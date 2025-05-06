#5-6变量作用域

# #局部变量
# def function1():
#     a="深圳" #局部变量
#     print(a)
# print(a)

#全局变量
# a="深圳"
# def function2():
#     a="上海" #相当于创建同名的局部变量
#     print(a)
#
# function2()
# print(a)

#global关键字
# a="深圳"
# def function2():
#     global a #声明a为全局变量
#     a="上海"
#     print(a)
#
# function2()
# print(a)

#嵌套作用域
def function3():
    a = "上海"
    print("function3被执行了！")
    def function4():
        print("function3中的function4被执行了！")
        print(a)

    function4()

function3()








