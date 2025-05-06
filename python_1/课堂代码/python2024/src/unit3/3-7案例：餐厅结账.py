# 3-7 案例：餐厅结账

# 为餐厅开发一个简易的结账系统(以3-5种菜品为例):
# 商品信息包含：
#       菜品编号 菜品价格 菜品名字  （使用不同变量保存）
# 1.提示用户输入菜品编号和数量,然后显示总价;
# 2.提示用户输入付款金额,进行核算。

num1="1001"
price1=29
name1="羊肉卷"

num2="1002"
price2=10
name2="千张"

num3="1003"
price3=5
name3="啤酒"

#提前声明变量保存价格和名称
price=0
name=""

num=input("请输入菜品编号:")
count=int(input("请输入购买数量:"))
if num==num1:
    price=price1
    name=name1
elif num==num2:
    price=price2
    name=name2
elif num==num3:
    price=price3
    name=name3
else:
    print("输入错误！")

amount=price*count
print("----您购买的是：",name,"单价：",price,"，数量：",count,"金额：",amount)

money=float(input("请输入付款金额："))
if money>=amount:
    print("----付款",money,"元！找零",money-amount,"元")
else:
    print("金额不足！")