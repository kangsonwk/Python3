# 9-3 json模拟数据库
import json

#读数据（查询）
def read():
    with open(r"users.txt","r")as f:
        jsonData=f.read()
    userList=json.loads(jsonData)
    return userList

#写数据（更新）
def write(userList):
    jsonData=json.dumps(userList,ensure_ascii=False)
    with open(r"users.txt","w")as f:
        f.write(jsonData)

#注册
def reg():
    uname = input("请输入新用户名：")
    upwd = input("请输入密码：")
    newUser={"uname":uname, "upwd": upwd} #新用户

    #验证
    userList = read()
    for user in userList:
        if uname == user["uname"]:
            print("用户名已存在！")
            return #终止函数
    #添加用户
    userList.append(newUser)
    write(userList)

#登录
def login():
    uname=input("请输入用户名：")
    upwd = input("请输入密码：")
    userList=read()
    msg="失败"
    for user in userList:
        if uname==user["uname"] and upwd==user["upwd"]:
            msg = "成功"
            print("-----登陆成功！")
    if msg=="失败":
        print("-----登陆失败！")
    return msg

def main():
    # #创建文本文件（数据库文件）
    # users='[{"uname":"zhangsan","upwd":"123"},{"uname":"lisi","upwd":"123"}]'
    # with open(r"users.txt","w")as f:
    #     f.write(users)
    reg()

if __name__ == '__main__':
    main()