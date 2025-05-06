#7-7
import os
import shutil

# #创建文件夹
# os.mkdir("hello")

# #删除文件夹
# if os.path.exists("hello"): #判断文件夹是否存在
#     shutil.rmtree("hello")

# #删除文件
# os.remove(r"C:\PycharmProjects\file\test2.txt")

# #移动文件或者文件夹
# shutil.move(r"C:\PycharmProjects\file\清晨.mp3",r"C:\PycharmProjects\file\aaa\清晨.mp3")

#查看文件夹信息
for files in os.walk(r"C:\PycharmProjects\file"):
    print(files[2])