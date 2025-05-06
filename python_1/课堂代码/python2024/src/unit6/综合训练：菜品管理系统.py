
#需求
# 1.使用列表嵌套字典的方式保存用户（用户名，密码，姓名，类型）；
#
# 2.使用列表嵌套字典的方式保存菜品（编号，名称，价格，库存）；
#
# 3.编写用户登录的函数，返回登录结果；
#
# 4.循环提示菜单，业务完毕时返回主菜单，退出时回到登陆页面；
#
# 5.将功能菜单中的业务功能各自编写到函数中；
#
# 6.用户选择不同业务编号时，调用已经写好的各种函数。


# 功能要点
# 1.用户登录；
# 2.显示菜品列表；
# 3.根据名称查询菜品信息；
# 4.菜品上下架（仅限管理员）；
# 5.设置菜品库存和价格（仅限管理员）；
# 6.库存预警（低于10 时）；
# 7.按照价格排序显示；


#数据准备
user1={"用户名":"aaa","密码":"123","姓名":"张三","类型":"店员"}
user2={"用户名":"bbb","密码":"123","姓名":"李四","类型":"店员"}
user3={"用户名":"ccc","密码":"123","姓名":"王五","类型":"管理员"}
usersList=[user1,user2,user3] #用户列表

p1={"编号":"1001","名称":"牛肉","价格":30,"库存":7,"更新人":"ccc"}
p2={"编号":"1002","名称":"千张","价格":10,"库存":122,"更新人":"ccc"}
p3={"编号":"1003","名称":"土豆","价格":10,"库存":103,"更新人":"ccc"}
p4={"编号":"1004","名称":"汽水","价格":6,"库存":34,"更新人":"ccc"}
p5={"编号":"1008","名称":"啤酒","价格":4,"库存":9,"更新人":"ccc"}
productsList=[p1,p2,p3,p4,p5] #菜品列表

currentUser={} #当前用户

#-------------功能函数

#验证管理员
def checkAdmin(fun):
    def wrapper():
        if currentUser["类型"]=="管理员":
            fun()
        else:
            print("没有权限！")
    return wrapper


# 1.用户登录；
def login():
    global currentUser
    result="失败"
    uname=input("请输入用户名：")
    upwd = input("请输入密码：")
    for user in usersList:
        if uname==user["用户名"] and upwd==user["密码"]:
            currentUser=user
            result="成功"
    return result

# 2.显示菜品列表；
def listAll():
    print("-编号----名称----价格----库存-")
    for product in productsList:
        print(product["编号"] + "-----" + product["名称"] + "-----" + str(product["价格"]) + "-------" + str(
            product["库存"]))

# 3.根据名称查询菜品信息；
def getOneByName():
    name=input("请输入要查询的菜品名称：")
    exist=0 #是否存在
    print("-编号----名称----价格----库存-")
    for product in productsList:
        if name==product["名称"]:
            exist=1
            print(product["编号"] + "-----" + product["名称"] + "-----" + str(product["价格"]) + "-------" + str(
                product["库存"]))
    if exist==0:
        print("未查询到该菜品！")

# 4.菜品上下架（仅限管理员）；
@checkAdmin
def add():
    #生成新编号
    listA=[]
    for p in productsList:
        listA.append(int(p["编号"]))
    newNum=str(max(listA)+1)
    uname=currentUser["用户名"]
    name=input("请输入菜品名称：")
    price = float(input("请输入菜品单价："))
    stock = int(input("请输入菜品库存："))
    newPro={"编号":newNum,"名称":name,"价格":price,"库存":stock,"更新人":uname}
    productsList.append(newPro)
    print(newPro["名称"], "上架成功！")
    listAll()
@checkAdmin
def delete():
    num=input("请输入要下架的菜品编号")
    exist=0 #菜品是否存在
    for p in productsList:
        if num==p["编号"]:
            exist=1
            productsList.remove(p) #删除
            print(p["名称"],"下架成功！")
    if exist==0:
        print("没有此菜品，下架失败！")
    listAll()

# 5.设置菜品库存和价格（仅限管理员）；
@checkAdmin
def update():
    while True:
        exist = 0  # 是否存在
        num=input("请输入要更新的菜品编号：")
        for product in productsList:
            if num== product["编号"]:
                exist=1
                input1=input("您要更新的是：1 价格 2 库存：")
                input2 = input("更新后的值为：")
                if input1=="1":
                    product["价格"]=float(input2)
                elif input1=="2":
                    product["库存"]=int(input2)
                print("--菜品",product["名称"],"已更新成功！")
                break
        if exist == 0:
            print("菜品不存在！")
            choice=input("取消请按1，重新输入请按2：")
            if choice=="1":
                break
            else:
                continue
        else:
            break

# 6.库存预警（低于10 时）；
def warning():
    exist = 0  # 是否存在
    print("-以下菜品需要补充食材-")
    print("-编号----名称----价格----库存-")
    for product in productsList:
        if product["库存"]<10:
            exist = 1
            print(product["编号"] + "-----" + product["名称"] + "-----" + str(product["价格"]) + "-------" + str(
                product["库存"]))
    if exist == 0:
        print("库存充足！")

# 7.按照价格排序显示；
def sortByPrice():
    choice=input("请选择升序或者降序（1.升序 2.降序）：")
    pList=[] #存放所有价格信息
    for product in productsList:
        pList.append(product["价格"])
    pList=list(set(pList)) #去掉重复价格
    if choice=="1":
        pList.sort()
    else:
        pList.sort(reverse=True)
    print("-编号----名称----价格----库存-")
    for price in pList:
        for product in productsList:
            if price == product["价格"]:
                print(product["编号"] + "-----" + product["名称"] + "-----" + str(product["价格"]) + "-------" + str(
                    product["库存"]))

#----------------显示主菜单，并调用相关功能
print("*************************** 51菜品管理系统 1.0 *******************************")
while True:
    result1=login()
    if result1=="失败":
        print("登录失败！请重新登录！")
        continue
    #业务功能
    while True:
        print("-------------------功能列表------------------")
        print("1.显示菜品列表;")
        print("2.根据名称查询菜品信息;")
        print("3.菜品上架;")
        print("4.菜品下架;")
        print("5.设置菜品库存和价格;")
        print("6.库存预警;")
        print("7.按照价格排序显示;")
        print("8.退出;")
        choice=input("---请输入功能编号(1-8)：")
        if choice=="1":
            listAll()
        elif choice=="2":
            getOneByName()
        elif choice=="3":
            add()
        elif choice=="4":
            delete()
        elif choice=="5":
            update()
        elif choice=="6":
            warning()
        elif choice=="7":
            sortByPrice()
        elif choice=="8":
            break
        else:
            print("没有此功能，请重新选择！")
            continue




