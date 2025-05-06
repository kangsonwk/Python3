#业务方法
import datetime

import dataBaseUtil

#登录
def login():
    uname=input("请输入用户名：")
    upwd=input("请输入密码：")
    sql="SELECT * FROM users WHERE name=(%s) and password=(%s)"
    users=dataBaseUtil.getData(sql, [uname,upwd])
    userInfo={} #用户信息
    if len(users)>0:
        userInfo = {"id":users[0][0],"name":users[0][1],"permission":users[0][3]}
        print("-----------登录成功！")
    else:
        print("-----------登录失败！")
    return userInfo

#1.查看菜单
def dishList():
    sql="SELECT * FROM dishes"
    dishes=dataBaseUtil.getData(sql, None)
    print("--id    名称    价格   折扣--")
    for dish in dishes:
        print(dish[0],"   ",dish[1],"   ",dish[2],"   ",dish[3])

#2.菜品上架
def addDish(userInfo):
    if userInfo["permission"]!="admin":
        print("--------无权限！")
        return
    name = input("请输入名称：")
    price = float(input("请输入价格："))
    cost = float(input("请输入成本："))
    #校验名称是否存在（省略）
    #新增
    sql="INSERT INTO dishes(name,price,discount,cost) VALUES((%s),(%s),(%s),(%s))"
    r=dataBaseUtil.updateData(sql, [name,price,1,cost])
    if r>0:
        print("菜品",name,"上架成功！")
    else:
        print("菜品", name, "上架失败！")

# 3.菜品下架
def deleteDish(userInfo):
    if userInfo["permission"]!="admin":
        print("--------无权限！")
        return
    name = input("请输入要下架的菜品名称：")
    sql="DELETE FROM dishes WHERE name=(%s)"
    r = dataBaseUtil.updateData(sql, [name])
    if r > 0:
        print("菜品", name, "下架成功！")
    else:
        print("菜品", name, "下架失败！")


# 4.菜品更新
def updateDish(userInfo):
    if userInfo["permission"]!="admin":
        print("--------无权限！")
        return
    name = input("请输入要修改的菜品名称：")
    c = input("请输入修改内容：1价格 2折扣：")
    sql = "UPDATE dishes"
    args=[]
    if c=="1":
        price = float(input("请输入新的价格："))
        sql="UPDATE dishes SET price=(%s) WHERE name=(%s)"
        args=[price,name]
    elif c == "2":
        discount = float(input("请输入新的折扣（0-1）："))
        sql="UPDATE dishes SET discount=(%s) WHERE name=(%s)"
        args = [discount, name]
    else:
        print("-----输入错误！")
        return
    r = dataBaseUtil.updateData(sql, args)
    if r > 0:
        print("菜品", name, "更新成功！")
    else:
        print("菜品", name, "更新失败！")

# 5.点菜 (加菜)
def orderDishes(userInfo):
    tableId = int(input("请输入餐桌编号："))
    while True:
        dishName=input("请输入菜品名称：")
        count = int(input("请输入数量："))
        sql1="SELECT * FROM dishes WHERE name=(%s)"
        dishes = dataBaseUtil.getData(sql1, [dishName])
        if len(dishes)<1:
            print("菜品不存在，请重新输入！")
            continue
        #新增订单
        amount=count*dishes[0][2]*dishes[0][3]
        createtime=datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        args=[count,amount,tableId,1,createtime,dishes[0][1],dishes[0][0],userInfo["id"],userInfo["name"]]
        sql2="INSERT INTO orders(count,amount,tableid,state,createtime,dishname,dishid,userid,username) VALUES((%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s))"
        r = dataBaseUtil.updateData(sql2, args)
        if r > 0:
            print("----", dishes[0][1], "已添加！")
        else:
            print("----", dishes[0][1], "添加失败！")
        c=input("输入1结束，输入其他内容继续：")
        if c=="1":
            break


# 7.删除订单
def deleteOrder(userInfo):
    if userInfo["permission"]!="admin":
        print("--------无权限！")
        return
    orderid = int(input("请输入要删除的订单id："))
    sql = "UPDATE orders SET state=(%s) WHERE id=(%s)" #逻辑删除
    args = [0, orderid]
    r = dataBaseUtil.updateData(sql, args)
    if r > 0:
        print("----订单已删除！")
    else:
        print("----订单删除失败！")

# 8.上菜
def serving():
    tableId = int(input("请输入餐桌编号："))
    sql="UPDATE orders SET state=2 WHERE tableid=(%s)"
    args = [tableId]
    r = dataBaseUtil.updateData(sql, args)
    if r > 0:
        print(tableId,"号桌----您的菜品已上齐！")
    else:
        print("----操作失败！")

# 9.结账
def checkout():
    tableId = int(input("请输入餐桌编号："))
    sql1 = "SELECT * FROM orders WHERE tableid=(%s) AND state=2"
    orders = dataBaseUtil.getData(sql1, [tableId])
    if len(orders) < 1:
        print("账已经结清！")
        return
    totalAmount=0
    print("--订单号    名称    数量   金额--")
    for order in orders:
        totalAmount=totalAmount+order[2]
        print(order[0],"  ",order[6],"  ",order[1],"  ",order[2])
    print("------------总消费金额：",totalAmount,"元！")
    c=input("确认结账请输入1:")
    if c=="1":
        sql2 = "UPDATE orders SET state=3 WHERE tableid=(%s) AND state=2"
        args = [tableId]
        r = dataBaseUtil.updateData(sql2, args)
        if r > 0:
            print(tableId, "号桌----已结账！")
        else:
            print(tableId, "号桌----结账失败！")

#10.订单统计
def orderStatistics(userInfo):
    if userInfo["permission"] != "admin":
        print("--------无权限！")
        return
    c = int(input("请输入(1.所有订单 2.按日期查询):"))
    if c==1:
        sql='''
        SELECT
            SUM(d.cost*o.count) AS total_cost,
            SUM(o.amount) AS total_expenditure,
            SUM(o.amount - d.cost*o.count) AS total_profit
        FROM
            orders o
        JOIN dishes d 
        ON o.dishid = d.id
        '''
        data = dataBaseUtil.getData(sql, [])
        print("汇总数据----总销售额：",data[0][1],"，总成本：",data[0][0],"，总利润：",data[0][2])
    elif c==2:
        date=input("请输入日期（格式2024/02/04）:")
        sql = '''
                SELECT
                    SUM(d.cost*o.count) AS total_cost,
                    SUM(o.amount) AS total_expenditure,
                    SUM(o.amount - d.cost*o.count) AS total_profit
                FROM
                    orders o
                JOIN dishes d 
                ON o.dishid = d.id
                WHERE
                DATE(o.createtime) = (%s);
                '''
        data = dataBaseUtil.getData(sql, [date])
        print(date,"----总销售额：", data[0][1], "，总成本：", data[0][0], "，总利润：", data[0][2])
    else:
        print("----输入错误！")




