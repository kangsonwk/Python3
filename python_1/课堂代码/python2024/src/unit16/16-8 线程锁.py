# 16-7 线程锁
import time
from threading import Thread,Lock
#当一个线程设置锁后，只有等到释放锁后，才能调度其他线程

#创建锁对象
lock=Lock()
num=100
def run(name):
    global num
    # lock.acquire() #加锁
    num=num-1
    print("线程",name,"执行了任务！目前num的值为",num)
    # lock.release() #解锁
    time.sleep(5)

def main():
    for i in range(1,101):
        t=Thread(target=run,args=(str(i),))
        t.start()

if __name__ == '__main__':
    main()