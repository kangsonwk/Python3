# tushare 财经数据接口包
# 安装 pip install tushare -i https://pypi.tuna.tsinghua.edu.cn/simple

# tushare股票价格自动监控
# 需求：设置一只股票卖出和买入价格，程序对价格进行监控，当价格达到预定值时发送邮件提醒---盯盘

import tushare
import time

def sendEmail():
    print("邮件发送中......")

def main():
    buyPoint=5.1
    salePoint=5.2
    n=0
    while True:
        data=tushare.get_realtime_quotes("601398")
        name=data.iloc[0]["name"] #名称
        price = float(data.iloc[0]["price"]) #当前价格
        pre_close = float(data.iloc[0]["pre_close"])  # 昨日收盘价
        change=round((price-pre_close)/pre_close,4)
        print("股票名称：",name," ，当前价格：",price,"，昨日收盘价格：",pre_close,"，涨幅：",change*100,"%")
        if price<=buyPoint and n!=1:
            print("股票达到买点，如果空仓请买进！")
            n=1
            sendEmail()
        elif price>=salePoint and n!=2:
            print("股票达到卖点，如果持仓请卖出！")
            n=2
            sendEmail()
        else:
            print("不要做任何交易！")
        time.sleep(5)





if __name__ == '__main__':
    main()