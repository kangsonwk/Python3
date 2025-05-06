#4-14冒泡排序

# #交换值
# a=3
# b=5
#
# c=a
# a=b
# b=c
#
# print(a,b)

listA=[54,32,45,27,18]

#升序排列
for i in range(1,len(listA)):
    print("第",i,"轮比较!")
    for j in range(0,len(listA)-i):
        print(j,"--",j+1)
        if listA[j]<listA[j+1]: #比较并交换值
            c=listA[j]
            listA[j]=listA[j+1]
            listA[j+1]=c
print(listA)

