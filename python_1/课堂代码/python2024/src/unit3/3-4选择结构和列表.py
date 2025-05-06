# 选择结构和列表

# 输入一个1-12的月份。
# 输出当前的季节。
# 3月到5月为春季    6月到8月为夏季
# 9月到11月为秋季   12月、1月和2月为冬季

month=int(input("请输入月份（1-12）："))
if month in [3,4,5]:
    print("春季！")
elif month in [6,7,8]:
    print("夏季！")
elif month in [9,10,11]:
    print("夏季！")
elif month in [12,1,2]:
    print("夏季！")
else:
    print("输入错误！")




# 在列表中保存餐厅菜单并展示。
# 支持用户增加或者减少菜品，
# 如果用户选择增加，则在列表中加上新菜品。
# 反之，在列表中减去用户输入的菜品。
# dishList=["番茄炒蛋","青椒肉丝","凉拌豆腐","回锅肉"]
# print("---------欢迎来到51餐厅，菜单：\n",dishList)
# choice=int(input("请输入功能编号（1.增加 2.删除）"))
# dishName=input("请输入菜品名称：")
# if choice==1:
#     dishList.append(dishName) #增加
# elif choice==2:
#     dishList.remove(dishName)  # 删除
# else:
#     print("没有此功能！")
# print("---------新菜单：\n",dishList)