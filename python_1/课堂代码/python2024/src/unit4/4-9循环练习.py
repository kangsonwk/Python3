# 4-9循环练习

# 1. 某农场有鸡兔同笼，上有35头，下有94足，问鸡兔各多少只？
# ji --- 0,35
# tu = 35-ji
# ji*2+tu*4=94
#穷举法
# for ji in range(0,36):
#     tu=35-ji #兔的数量
#     if ji*2+tu*4==94:
#         print("鸡的数量：",ji,"兔的数量：",tu)
#         break

# 2.一张纸的厚度大约是0.08mm，对折多少次之后能达到珠穆朗玛峰的高度（8844米）？
# p=0.08
# m=8844000
# count=0
# while p<m:
#     p=p*2 #对折
#     count=count+1
#     print("当前纸的厚度为：",p)
# print("对折次数：",count)


# # 3.输入任意个学员的分数，获取最高分、最低分、平均分。
# count=0
# maxScore=0
# minScore=0
# totalScore=0
# while True:
#     count = count + 1
#     score=int(input("请输入第"+str(count)+"个分数："))
#     if count==1: #第一个分数
#         maxScore=score
#         minScore=score
#     if score>maxScore:
#         maxScore=score
#     if score<minScore:
#         minScore=score
#     totalScore=totalScore+score
#     choice=input("结束请输入1，继续请输入其他值：")
#     if choice=="1":
#         break
# print("最高分：",maxScore,"最低分：",minScore,"平均分：",totalScore/count)


# 4.修改餐厅结账程序，实现重复添加多个菜品；
num1="1001"
price1=29
name1="羊肉卷"

num2="1002"
price2=10
name2="千张"

num3="1003"
price3=5
name3="啤酒"

totalAmount=0 #总金额
totalcount=0 #总数量

while True:
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
    amount=price*count #单个菜品金额
    totalAmount=totalAmount+amount
    totalcount=totalcount+count
    print("----您当前添加的是：",name,"单价：",price,"，数量：",count,"金额：",amount)
    choice = input("结束请输入1，继续请输入其他值：")
    if choice=="1":
        break
print("======您本次共消费",totalAmount,"元！菜品数量:",totalcount)