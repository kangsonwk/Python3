
#7-5文件路径和编码
# file1=open(r"file/abc.txt","r")
# data1=file1.read()
# file1.close()
#
# print(data1)

#gbk utf-8 gb2312 utf16
file2=open(r"../../abc.txt","r",encoding="gbk")
data2=file2.read()
file2.close()

print(data2)
