# 2-11列表
#特性：有序;元素值可以重复；可以存放多种数据类型
lista=["张三","李四","王五","赵六","赵六",666]
#        0     1     2     3      4
#              -4    -3    -2    -1

#通过索引（下标）获取值
# print(lista[4])
# print(lista[-1])


#切片
listb=[12,34,56,78,90,13,36,67,78,89]
#       0  1  2  3  4  5  6  7  8  9

# print(listb[2:8:1]) #(listb[起始索引:结束索引（不包含）:步长]
# print(listb[4:]) #获取某个索引以后的所有值
# print(listb[:4]) #获取某个索引以前的所有值

#增加数据
lista=["张三","李四","王五","赵六"]
lista.append("大宝") #在末尾添加元素
lista.insert(2,"乔峰") #在指定位置添加元素
# print(lista[3])

#删除数据
lista=["张三","李四","王五","赵六"]
lista.remove("王五") #删除具体值
lista.pop(2)
# print(lista)

#给元素重新赋值
lista=["张三","李四","王五","赵六"]
lista[0]="张三丰"
# print(lista)

#列表合并
lista=["张三","李四","王五","赵六"]
listb=["大宝","二宝","小宝"]
lists=lista+listb
print(lists)






