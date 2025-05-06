# 16-6 多线程实现
import time

#引入线程类
from threading import Thread

#任务
def run(name):
    print(name,"执行了任务")
    time.sleep(5)

def main():
    # -------并行 创建线程对象
    t1=Thread(target=run,args=("线程1",)) #参数：目标方法（函数名），参数列表
    t2 = Thread(target=run,args=("线程2",))
    t3 = Thread(target=run,args=("线程3",))

    #启动线程
    t1.start()
    t2.start()
    t3.start()


if __name__ == '__main__':
    main()