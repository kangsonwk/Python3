# 4-7循环嵌套
# day=1
# while day<=7: #外层循环
#     print("第",day,"天到了！")
#     i=1
#     while i<=10: #内层循环
#         print("第",day,"天，第",i,"次打印：你好，世界！")
#         i=i+1
#     day=day+1


# #还款
# totalMoney=0 #累计还款金额
# for year in range(1,11):
#     print("-------第",year,"年到了！")
#     for month in range(1,13):
#         totalMoney=totalMoney+1
#         print("第",year,"年，第",month,"月，还款1万元，累计已还",totalMoney,"万元！还剩",120-totalMoney,"万元！")

#遍历多维容器
list1=[21,2,335,5,446,57,57]
list2=[22,35,4,6,656,57,77,14]
list3=[4,65,767,86,7]
listX=[list1,list2,list3]
for i in listX:
    print("--------------------")
    for j in i:
        print(j)