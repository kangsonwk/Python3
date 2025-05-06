#7-4读写二进制文件

file1=open(r"C:\PycharmProjects\file\cat.jpg","rb")
data1=file1.read()
file1.close()
print(data1)

file2=open(r"C:\PycharmProjects\file\aaa\cat2.jpg","wb")
file2.write(data1)
file2.close()

file3=open(r"C:\PycharmProjects\file\aaa\cat2.jpg","ab")
file3.write(data1)
file3.close()


