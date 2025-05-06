# 4-2 求和问题

# #1-100之间所有整数之和
# num=0
# for i in range(1,101):
#     num=num+i
# print(num)


#还款
totalMoney=0 #累计还款金额
for i in range(1,11):
    totalMoney=totalMoney+12
    print("第",i,"年，还款12万元！累计已还",totalMoney,"万！，还差",120-totalMoney,"万！")
print("---已还清",totalMoney,"万元！")


