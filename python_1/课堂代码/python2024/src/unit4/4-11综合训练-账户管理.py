# 4-11账户管理

# 1. 模拟3个账户,1001,1002,1003,分别设置密码和余额(使用列表嵌套字典的方式)；
# 2. 提示用户输入账号和密码，遍历每张卡的信息验证是否成功；
# 3. 如果用户输入正确---提示让用户选择查询、转账、充值、还是退出,并提示余额.
#    输入错误---重新输入账号密码；
# 选择转账---输入转账金额和账号,如超过余额,提示余额不足;否则,在余额上减掉相应金额；
# 选择充值---输入充值金额,提示充值后余额；
# 选择退出---重新登录；
# 4. 设置3次输入错误账号密码,提示账户已被锁定，程序结束。

account1={"姓名":"张三","账号":"1001","密码":"123","余额":10000}
account2={"姓名":"李四","账号":"1002","密码":"123","余额":20000}
account3={"姓名":"王五","账号":"1003","密码":"123","余额":30000}
acList=[account1,account2,account3]
count=0 #记录登录失败的次数
while True:
    unum=input("请输入账户名：")
    upwd = input("请输入密码：")
    msg=0 #记录登录状态 0未登录 1登陆成功
    thisAccount={} #保存当前账户
    for account in acList:
        if unum==account["账号"] and upwd==account["密码"]:
            msg=1
            thisAccount=account
            break #终止for
    if msg==1:
        print("登陆成功！你好，",thisAccount["姓名"],"！")
        count = 0 #登录失败次数清零
        # break
    else:
        count=count+1
        if count>=3:
            print("您已经连续三次输入错误！账户已被锁定！")
            break
        else:
            print("登录失败！您已经连续",count,"次输入错误，还有",3-count,"次机会！")
            continue

    #具体业务
    while True:
        choice=int(input("请输入要办理的业务编号（0.查询 1.转账 2.充值 3.退出）："))
        if choice==0:
            print(thisAccount)
        elif choice==1:
            ac=input("请输入转账账户：")
            money1 = float(input("请输入转账金额："))
            #验证账户
            msg2=0 #记录账户是否存在 0不存在
            for account in acList:
                if ac==account["账号"]:
                    msg2=1
                    break
            if msg2==0:
                print("账户不存在！请重新选择业务编号！")
                continue
            #验证金额
            if money1>thisAccount["余额"]:
                print("余额不足！请重新选择业务编号！")
                continue
            else:
                thisAccount["余额"]=thisAccount["余额"]-money1
                print("转账成功！转出",money1,"元！余额",thisAccount["余额"],"元！")
        elif choice==2:
            money2 = float(input("请输入充值金额："))
            if money2<=0:
                print("充值金额必须大于0！请重新选择业务编号！")
                continue
            else:
                thisAccount["余额"] = thisAccount["余额"] + money2
                print("充值成功！充值", money2, "元！余额", thisAccount["余额"], "元！")

        elif choice==3:
            break
        else:
            print("没有此功能，请重新选择！")
