#8-4内置模块-time
import time

def main():
    # #获取时间戳 现在-1970
    # s1=time.time()
    # for i in range(1,10001):
    #     print("次数：",i)
    # s2 = time.time()
    # print("运行时间：",s2-s1)

    # #获取日期时间
    # t=time.localtime()
    # print(time.asctime(t))#格式化
    #
    # #将日期时间转换成指定格式 2024年1月19日 14:01:22
    # strTime=time.strftime("%Y年%m月%d日 %H:%M:%S",t)
    # print(strTime)

    # #程序休眠
    # print("程序开始！")
    # time.sleep(5)
    # print("程序结束！")

    # #倒计时
    # for i in range(10,-1,-1):
    #     print(i)
    #     time.sleep(1)

    time.sleep(3*24*60*60)

if __name__ == '__main__':
    main()






