# 2-9接收键盘输入

# 接收用户输入数据的变量=input("提示语句")
#提示语句必须是一个单独的字符串（一个参数）
name=input("请输入你的姓名：")
# print("你好呀！"+name)

#input接收到的数据统一是字符串
money=int(input("请输入你的银行存款："))
print(name+",你的银行存款是",money)
print("明年存款变成：",money*1.03)

