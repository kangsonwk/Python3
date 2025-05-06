# 11-6 案例：银行卡

class Card:
    def __init__(self,cnum,cpwd,cname,cbalance):
        self.cnum=cnum
        self.cpwd = cpwd
        self.cname = cname
        self.cbalance = cbalance
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
        print("您的余额是：",self.cbalance)

    #存款
    def deposit(self):
        r=self.login() #在类的内部调用类的其他方法
        if r=="ok":
            money=float(input("请输入存款金额："))
            self.cbalance=self.cbalance+money
            print("存款成功!存入",money,"元！余额",self.cbalance,"元！")


#----------------------------------------------
def main():
    card1=Card("1001","123","zhangsan",1000)
    card2 = Card("1002", "123", "lisi", 2000)

    card2.deposit()
    card2.show()

if __name__ == '__main__':
    main()