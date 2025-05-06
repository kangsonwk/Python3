# 定义类

#猫类  class 类名:
class Cat:
    #初始化方法  self.属性名 定义属性
    def __init__(self,nick,color,age):
        #属性：昵称、颜色、年龄
        self.nick=nick
        self.color=color
        self.age=age
    def eat(self):
        print("猫在吃鱼！")
    def sleep(self):
        print("猫在睡觉！")


#图书类
class Book:
    def __init__(self,id,name,author):
        self.id=id
        self.name = name
        self.author = author
    def show(self):
        print("显示图书详情！")
    def update(self):
        print("更新图书信息！")




def main():
    pass

if __name__ == '__main__':
    main()