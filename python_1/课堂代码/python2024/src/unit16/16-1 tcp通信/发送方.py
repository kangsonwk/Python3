#发送消息给接收方
from socket import *

def main():
    #创建socket对象
    #参数1：地址族 AF_INET（ipv4）AF_INET6（ipv6） AF_UNIX (本机通信)
    #参数2：套接字协议类型 SOCK_STREAM（TCP）  SOCK_DGRAM（UDP）
    s=socket(AF_INET,SOCK_STREAM)
    try:
        #建立连接
        s.connect(("localhost",6363))
        #发送消息
        s.send("你好！张三！".encode()) #.encode()对消息进行编码
    except Exception as e:
        print("出现异常",e)
    finally:
        #关闭socket
        s.close()

if __name__ == '__main__':
    main()