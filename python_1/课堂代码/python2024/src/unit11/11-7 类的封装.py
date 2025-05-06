# 11-7 类的封装

class Card:
    def __init__(self,cnum,cpwd,cname,cbalance):
        self.cnum=cnum
        self.cpwd = cpwd
        self.cname = cname
        self.__cbalance = cbalance #私有属性-只能在类的内部被访问
        self.bank="开发银行"
    #登录
    def login(self):
        num=input("请输入卡号：")
        pwd = input("请输入密码：")
        if num==self.cnum and pwd==self.cpwd:
            print("验证成功!")
            return "ok"
        else:
            print("验证失败！")
            return "no"
    #显示余额
    def show(self):
        r = self.login()  # 在类的内部调用类的其他方法
        if r == "ok":
            print("您的余额是：",self.__cbalance)
    #存款
    def deposit(self):
        r=self.login() #在类的内部调用类的其他方法
        if r=="ok":
            money=float(input("请输入存款金额："))
            if money>=100000:
                self.__reword()
            self.__cbalance=self.__cbalance+money
            print("存款成功!存入",money,"元！余额",self.__cbalance,"元！")
    #奖励
    def __reword(self): #私有方法
        print("奖励手机一部！")


#----------------------------------------------
def main():
    card1=Card("1001","123","zhangsan",1000)
    card2 = Card("1002", "123", "lisi", 2000)

    card1.deposit()

if __name__ == '__main__':
    main()