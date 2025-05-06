#7-3写文件

# w写文件 write
#如果文件存在，则打开并写入；如果文件不存在，则创建新文件并写入
#文件夹路径必须存在
#如果文件中有内容，则先清空
s="你好，深圳！"
file1=open(r"C:\PycharmProjects\file\test1.txt","w")
file1.write(s)
file1.close()


# a追加写文件 append
# #如果文件存在，则打开并追加写入；如果文件不存在，则创建新文件并写入
# #文件夹路径必须存在
# s="你好，北京！"
# file1=open(r"C:\PycharmProjects\file\test1.txt","a")
# file1.write(s)
# file1.close()
