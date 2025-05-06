# 8-6第三方模块例子
#pillow 图片处理模块
from PIL import Image

def main():
    img=Image.open(r"C:\PycharmProjects\file\cat.jpg") #读取图片
    newImg=img.convert("L")   #L黑白
    newImg.save(r"C:\PycharmProjects\file\cat2.jpg")


if __name__ == '__main__':
    main()
