# 3-5选择结构练习

# 1.判断奇偶数
# 根据用户输入的整数，判断是偶数还是奇数：
# s1=input("请输入一个数：")
# print(s1.isdigit()) #判断这个字符串是否是数值形态
# if s1.isdigit():
#     num=int()
#     if num%2==0:
#         print("偶数")
#     else:
#         print("奇数")
# else:
#     print("输入的不是数字！")




# 2.用户登录
# 用户输入账号密码，判断是否登陆成功。
# name="zhangsan"
# pwd="12345a"
#
# uName=input("请输入用户名：")
# uPwd=input("请输入密码：")
#
# if uName==name and uPwd==pwd:
#     print("登录成功！")
# else:
#     print("登录失败！")




# 3.打开保险柜
#
# 现有一个银行保险柜，有两道密码。
# 想拿到里面的钱必须两次输入的密码都要正确。
#
# 如果第一道密码都不正确，那直接把你拦在外面；
#
# 如果第一道密码输入正确，才能有权输入第二道密码。
#
# 只有当第二道密码也输入正确，才能拿到钱！(两道密码自己设置)

# pwd1="123a"
# pwd2="12345a"
# uPwd1=input("请输入第一道密码：")
# if uPwd1==pwd1:
#     uPwd2=input("请输入第二道密码：")
#     if uPwd2==pwd2:
#         print("恭喜你拿到钱：5毛！")
#     else:
#         print("抱歉，密码错误！")
# else:
#     print("你出去吧！")


# 4.煎饼果子计算器
#
# 设计一个程序，根据用户选择的煎饼类型、配料和数量，计算出价格。
#
# 用户选择煎饼类型
# （小份 5元，中份 6元，大份 7元）；
#
# 用户选择配料
# （鸡蛋 2元、蔬菜 3元、火腿 4元）；
#
# 如果满10元，送豆浆一杯。
type=int(input("你要什么类型煎饼(1.小份 2.中份 3.大份)："))
mixture=int(input("你要什么配菜(1.鸡蛋 2.蔬菜 3.火腿)："))
num=int(input("你要几个？"))
amount=0 #总金额

#类型的判断
if type==1:
    amount=amount+5
elif type==2:
    amount=amount+6
elif type==3:
    amount=amount+7
else:
    print("选择错误！")

#配菜的判断
if mixture==1:
    amount=amount+2
elif mixture==2:
    amount=amount+3
elif mixture==3:
    amount=amount+4
else:
    print("选择错误！")
amount=amount*num #乘以数量获取总金额
print("您的订单总金额为",amount,"元！")

if amount>=10:
    print("赠送豆浆一杯！")