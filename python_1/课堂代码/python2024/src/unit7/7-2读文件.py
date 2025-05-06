#7-2读文件

#"r" 读文本文件 read
# file=open(r"C:\PycharmProjects\file\再别康桥.txt","r") #打开或创建文件
# data=file.read() #读取文件
# # data=file.readline() #按行读取文件，每次只读一行
# # data=file.readlines() #按行读取文件并存入列表
# print(data)
# file.close()#关闭文件资源

#文件指针
file=open(r"C:\PycharmProjects\file\abc.txt","r") #打开或创建文件
data1=file.read()
file.seek(0) #将指针手动定位到开头
data2=file.read()
print(data1)
print("------------------------------")
print(data2)
file.close()#关闭文件资源
