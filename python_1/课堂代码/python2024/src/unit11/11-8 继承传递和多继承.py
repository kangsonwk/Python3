# 11-9 继承传递和多继承
class Animal: #父类
    def __init__(self,nick,color,age):
        self.nick=nick
        self.color = color
        self.age = age
    def run(self):
        print("动物在跑！")
    def eat(self):
        print("动物在吃！")

class Pet: #父类
    def __init__(self,strain):
        self.strain=strain
    def play(self):
        print("宠物在玩耍！")


class Dog(Animal): #一个类既可以是父类，也可以是子类
    def run(self):
        print("狗在飞快的跑！")

#继承传递
class Ha(Dog):
    pass

#多继承
#多个父类拥有同一个方法时，按照继承顺序调用
class Cat(Pet,Animal):
    pass

#----------------------------------------------
def main():
    # ha=Ha("小白", "白色", 2) #继承是可以传递
    # ha.run()

    cat=Cat()

if __name__ == '__main__':
    main()