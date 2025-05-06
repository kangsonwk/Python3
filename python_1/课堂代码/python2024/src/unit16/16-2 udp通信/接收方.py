#接收发送方的信息

from socket import *

def main():
    # 创建socket对象
    s = socket(AF_INET, SOCK_DGRAM)

    #绑定监听端口
    s.bind(("localhost",6364))

    #接收消息
    while True:
        msg=s.recv(1024) #消息最大字节数
        print("-------------",msg.decode()) #解码并输出

if __name__ == '__main__':
    main()
