# 11-5 self关键字和初始化方法

#self:当前对象
#初始化方法：创建对象时调用

class Cat:
    #初始化方法  self.属性名 定义属性
    def __init__(self,nick,color,age):
        print("Cat类中",nick,"对象被创建了！")
        #属性：昵称、颜色、年龄
        self.nick=nick
        self.color=color
        self.age=age

    def eat(self,num): #self.属性名  self会隐式传递当前对象
        print(self.nick,"在吃鱼！吃了",num,"条！")

    def sleep(self):
        print("猫在睡觉！")

#----------------------------------------------
def main():
    #对象名=类名(属性值1，属性值2，属性值3)
    cat1=Cat("小白","白色",2)
    cat2=Cat("小黑", "黑色", 3)

    # cat1.eat(3)
    # cat2.eat(3)




if __name__ == '__main__':
    main()