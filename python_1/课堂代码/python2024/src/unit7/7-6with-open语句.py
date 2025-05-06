#7-6with-open语句
# with open(r"C:\PycharmProjects\file\再别康桥.txt","r") as file:
#     data=file.read()
# print(data)

with open(r"C:\PycharmProjects\file\test1.txt","r") as file1,open(r"C:\PycharmProjects\file\aaa\test1.txt","w") as file2:
    data=file1.read()
    file2.write(data)




