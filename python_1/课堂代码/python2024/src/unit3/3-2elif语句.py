# 3-2elif语句
money=int(input("请输入奖金金额（元）:"))
if money>8000:
    print("吃大龙虾！真开心！")
elif money>5000:
    print("吃火锅！也不错！")
elif money>3000:
    print("吃鸡公煲！还能接受！")
else:
    print("吃热干面！")

#if在开头，不可省略
#elif可以有多个
#else只有末尾的一个，可省略
#多条件判断中，只执行第一个满足条件的语句




