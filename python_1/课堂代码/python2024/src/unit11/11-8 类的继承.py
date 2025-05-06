# 11-8 类的继承

# 继承：子类继承父类，子类将自动拥有父类的属性和方法
#一个父类可以有多个子类，一个子类也可以有多个父类（多继承）
#作用：将多个类中共同的属性和方法定义到父类中，减少代码冗余
class Animal: #父类
    def __init__(self,nick,color,age):
        self.nick=nick
        self.color = color
        self.age = age
    def run(self):
        print("动物在跑！")
    def eat(self):
        print("动物在吃！")

class Cat(Animal): #子类 继承Animal
    def run(self):#重写
        print("猫在慢慢地走！")
    def sleep(self):
        print("猫在睡觉！")

class Dog(Animal):
    def run(self):
        print("狗在飞快的跑！")


#----------------------------------------------
def main():
    cat=Cat("小白","白色",2)
    dog = Dog("小白", "白色", 2)
    cat.run()
    dog.run()

if __name__ == '__main__':
    main()