# 4-4循环和选择结构组合

# #判断某个名字是否存在
# names=["张三","李四","王五","赵六"]
# key=input("请输入要查询的姓名：")
# result="不存在" #保存结果
# for name in names:
#     if name==key:
#         result="存在"
# print(result)

# #计算及格率
# scores=(67,78,68,87,92,45,69,77,53,89)
# count=0
# for score in scores:
#     if score>=60:
#         count=count+1
# print(count/len(scores))


#苹果打8折，其他商品不打折，求购物车总价
cart = {"apple": 25, "banana": 12, "orange": 9}
totalPrice=0
for key in cart:
    name=key
    price=cart[key]
    if name=="apple":
        price=price*0.8
    totalPrice=totalPrice+price
print("购物车总价为：",totalPrice)
