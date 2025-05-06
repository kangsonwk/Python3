#8-3内置模块-random
import random


def main():
    r1=random.randint(1,6) #生成指定的随机整数
    r2=random.uniform(1,3) #生成随机浮点数
    lista = ["张三", "李四", "王五", "赵六", "赵六"]
    name=random.choice(lista)
    # print(name)

    # 抽奖程序：
    # 一等奖 笔记本电脑 0.1%概率
    # 二等奖 冰箱 1%概率
    # 三等奖 音响 10%概率
    # 谢谢惠顾
    #中奖号码：1-1000
    r = random.randint(1, 1000)
    if r==1:
        print("恭喜！获得笔记本一台！一等奖！")
    elif 11<=r<=20:
        print("恭喜！获得冰箱一台！二等奖！")
    elif 101<=r<=200:
        print("恭喜！获得音响一台！三等奖！")
    else:
        print("谢谢惠顾！")

if __name__ == '__main__':
    main()






