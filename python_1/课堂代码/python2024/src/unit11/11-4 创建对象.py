# 创建对象

class Cat:
    #初始化方法  self.属性名 定义属性
    def __init__(self,nick,color,age):
        #属性：昵称、颜色、年龄
        self.nick=nick
        self.color=color
        self.age=age
    def eat(self,num):
        print("猫在吃鱼！吃了",num,"条！")
        return "骨头"
    def sleep(self):
        print("猫在睡觉！")

#----------------------------------------------
def main():
    #对象名=类名(属性值1，属性值2，属性值3)
    cat1=Cat("小白","白色",2)
    cat2 = Cat("小黑", "黑色", 3)

    #获取属性值  对象名.属性名
    cat2.nick="小小黑" #给属性重新赋值
    print(cat2.nick)

    #调用对象方法  对象名.方法名()
    result=cat1.eat(3)
    print(result)
    cat2.sleep()

if __name__ == '__main__':
    main()