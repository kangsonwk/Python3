# while循环
# i=1
# while i<=10:
#     print("第",i,"次打印：你好，世界！")
#     i = i + 1

#for循环适合循环次数确定的业务，可以用来遍历容器
#while适合已知循环执行条件的业务


# 用户输入账号密码，判断是否登陆成功。 允许多次输入直到登陆成功
name="zhangsan"
pwd="12345a"
state="失败"
while state=="失败":
    uName = input("请输入用户名：")
    uPwd = input("请输入密码：")
    if uName == name and uPwd == pwd:
        state = "成功"
        print("登录成功！")
    else:
        print("登录失败！")







