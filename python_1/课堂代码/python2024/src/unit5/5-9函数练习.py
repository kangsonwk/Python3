#5-9函数练习

# 1.写函数，接收3个数字参数，返回最大的那个数字。
# def getMax(a,b,c):
#     listA=[a,b,c]
#     return max(listA)
# print(getMax(23,45,67))

# # 2.输入一个列表，返回手动反转后的新列表。
# def getReverse(listAttr):
#     resultList = []
#     for i in range(len(listAttr) - 1, -1, -1):  # 逆序遍历
#         resultList.append(listAttr[i])
#     return resultList
#
# listA=[12,23,45,32,16]
# listB=getReverse(listA)
# print(listB)



# # 3.编写一个用户登录函数（用户名密码提前设置）， 返回用户登录成功或者失败的结果。
# name="zhangsan"
# pwd="123"
#
# def login(uname,upwd):
#     if uname==name and upwd==pwd:
#         return 1
#     else:
#         return 0
# print(login("zhangsan","123"))


# # 4.设计一个函数来计算商品的总价格，该价格等于商品数量乘以单价。
# # 如果商品数量大于100，享受10%的折扣。数量在50到100之间，享受5%的折扣。
# # 数量小于50，没有折扣。
# def getAmount(num,price):
#     amount=num*price
#     if num>100:
#         amount=amount*0.9
#     elif num>50:
#         amount = amount * 0.95
#     return amount
#
# result=getAmount(20,100)
# print(result)



# # 5.设计一个函数来计算存款利息。存款利息由存款金额和存款时间决定。
# #  如果存款金额小于或等于5000元，则年利率为2%；
# #  5000元到10000元之间，则年利率为3%；大于10000元，则年利率为4%。
# #  返回本金和利息。
# def getMoney(amount,year):
#     rate=0 #利率
#     if amount>10000:
#         rate=0.04
#     elif amount>5000:
#         rate = 0.03
#     else:
#         rate = 0.02
#     totalAmount=amount*((1+rate)**year)
#     return  totalAmount,amount,totalAmount-amount
# print(getMoney(20000,3))

# 6.编写一个函数，接收一个字符串参数，将其中的敏感词替换为星号，并返回替换后的结果。
def replaceWords(s):
    wordsList=["阿里巴巴","苹果","亚马逊","京东","字节","脸书"]
    for word in wordsList:
        s=s.replace(word,"*"*len(word))
    return s
print(replaceWords("阿里巴巴、亚马逊、字节和脸书是世界上最知名的公司之一"))

