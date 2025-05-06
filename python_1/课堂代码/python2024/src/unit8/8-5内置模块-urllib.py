#8-5内置模块-urllib
from urllib import request

def main():
    url="http://www.baidu.com"
    data=request.urlopen(url).read() #发送请求并读取响应信息
    print(data.decode()) #.decode() 解码：将二进制转换成普通字符


if __name__ == '__main__':
    main()






