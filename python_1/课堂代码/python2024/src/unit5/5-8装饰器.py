#5-8装饰器

# #定义装饰器
# def decorate(func): #接收函数作为参数
#     def wrapper():
#         print("开始调用函数...")
#         func()
#         print("函数调用结束...")
#     return wrapper
#
# @decorate  #注解：给函数附加一些其他功能
# def sayHello():
#     print("你好，世界！")
#
# sayHello()

#定义装饰器
def decorate2(func): #接收函数作为参数
    def wrapper():
        inn=input("请输入用户口令：")
        if inn=="12345":
            print("用户验证成功!")
            func() #执行目标函数
        else:
            print("用户验证失败！")
    return wrapper

@decorate2  #注解：给函数附加一些其他功能
def transfer():
    print("转账中.........")

transfer()