#8-5内置模块-datetime
import datetime

def main():
    # 获取当前的日期时间
    now = datetime.datetime.now()
    print(now)

    # 日期转字符串
    s1 = now.strftime("%Y年-%m月-%d日  %H:%M:%S")
    print(s1)

    # 字符串转日期(格式一定要一致)
    s2 = "2018/02/13 05:23:45"
    d2 = datetime.datetime.strptime(s2, "%Y/%m/%d %H:%M:%S")
    print(d2)

    #获取指定时间
    dt=datetime.datetime(2023,11,23,11,12,13)
    print(dt)

    #时间计算
    newDt1=dt-datetime.timedelta(days=1)
    newDt2 = dt - datetime.timedelta(hours=2)
    newDt3 = dt + datetime.timedelta(minutes=30)
    print(newDt3-newDt2)



if __name__ == '__main__':
    main()






