# 3-3选择结构嵌套

# money=int(input("请输入奖金金额（元）:"))
# day=input("请输入今天星期几（1-7）:")
# if money>8000:
#     print("吃大龙虾！真开心！今天就吃！")
# elif money>5000:
#     print("吃火锅！也不错！")
#     if day=="6" or day=="7":
#         print("今天就去！")
#     else:
#         print("周末再去！")
# elif money>3000:
#     print("吃鸡公煲！还能接受！今天就吃！")
# else:
#     print("吃热干面！今天就吃！")

amount=int(input("请输入消费金额（元）:"))
isVip=input("请输入是否vip（是/否）:")
if amount>=100:
    if isVip=="是":
        print("你是vip,消费" , amount*0.8 , "元！")
    else:
        print("你不是vip,消费" , amount * 0.9 , "元！")
else:
    print("消费",amount,"元！")


