#7-8文件读写练习
import os
import shutil

# # 1. 文档《人物介绍》中姓名错了，请使用程序
# # 将其中的“刘小言”更正成“刘晓言”
# with open(r"C:\PycharmProjects\file\人物介绍.txt","r",encoding="utf-8") as file1:
#     data1=file1.read()
# newData1=data1.replace("刘小言","刘晓言")
# with open(r"C:\PycharmProjects\file\人物介绍.txt","w") as file2:
#     file2.write(newData1)

# # 2.使用文本文件制作一个简易的记账本，
# # 功能包含添加账目和查看流水，格式参照文档
# # 《记账本》。
# #支出---时间：2024.1.18  支出原因 ：午饭     金额：12
# choice=input("请选择：1 增加账目 2 查看流水：")
# if choice=="1":
#     time=input("请输入时间：")
#     s = input("请输入收入或者支出：")
#     amount = input("请输入金额：")
#     reason = input("请输入原因：")
#     strr="\n"+s+"---时间："+time+" 原因 ："+reason+" 金额："+amount
#     with open(r"C:\PycharmProjects\file\记账本.txt","a",encoding="gbk") as file1:
#         file1.write(strr)
# else:
#     with open(r"C:\PycharmProjects\file\记账本.txt", "r",encoding="gbk") as file2:
#         data2=file2.read()
#     print(data2)


# # 3.给记账本增加一个统计功能，统计总开支
# # 和总收入。
# # with open(r"C:\PycharmProjects\file\记账本.txt","a",encoding="gbk") as file1:
# #     file1.write("\n收入---时间：2024.1.18  支出原因 ：午饭 金额：15")
# with open(r"C:\PycharmProjects\file\记账本.txt", "r",encoding="gbk") as file2:
#         dataList=file2.readlines()
# inn=0
# out=0
# for data in dataList:
#     data=data.replace("\n","") #去掉换行符
#     list1=data.split("---")
#     key1=list1[0]
#     list2 = data.split("金额：")
#     key2 = float(list2[1])
#     if key1=="收入":
#         inn=inn+key2
#     else:
#         out = inn + key2
# print("总收入：",inn,"总支出：",out)


# 4.对文档《美食大全》中各个美食介绍进行分割，将分割后的文档存到“/美食大全”目录。
#C:\PycharmProjects\file\美食大全
with open(r"C:\PycharmProjects\file\美食大全.txt", "r", encoding="utf-8") as file3:
    data3 = file3.read()
list2 = data3.split("*****")
i=0
for txt in list2:
    i=i+1
    fileName=r"\美食大全"+str(i)+".txt"
    with open(r"C:\PycharmProjects\file\美食大全"+fileName, "w", encoding="utf-8") as file4:
        file4.write(txt)