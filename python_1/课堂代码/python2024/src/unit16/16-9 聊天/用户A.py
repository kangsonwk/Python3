from socket import *
from threading import Thread

isInput=0 #0不输入 1输入

#接收消息
def recvData(soc):
    global isInput
    while True:
        msg=soc.recv(1024)
        print(">>:",msg.decode())
        isInput=1

#发送消息
def sendData(soc,ip,port):
    global isInput
    while True:
        if isInput==1:
            info=input("<<:")
            soc.sendto(info.encode(),(ip,port))
            isInput=0

def main():
    ip="localhost" #对方ip
    port=9011#对方端口号

    # 创建socket对象
    s = socket(AF_INET, SOCK_DGRAM)
    # 绑定监听端口
    s.bind(("localhost", 9012))

    #创建线程
    t1=Thread(target=recvData,args=(s,))
    t2 = Thread(target=sendData, args=(s,ip,port,))
    t1.start()
    t2.start()

if __name__ == '__main__':
    main()