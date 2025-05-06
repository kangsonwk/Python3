# 16-5 多进程实现
import time
#一个程序运行过程中，会产生多少个进程？---至少一个

#引入进程类
from multiprocessing import Process

#任务1
def run1():
    print("执行了任务1")
    time.sleep(5)
#任务2
def run2():
    print("执行了任务2")
    time.sleep(5)
#任务3
def run3(name):
    print(name,"执行了任务3")
    time.sleep(5)

def main():
    # #-------串行
    # run1()
    # run2()
    # run3()

    # -------并行 创建进程对象
    p1=Process(target=run1) #参数：目标方法（函数名），参数列表
    p2 = Process(target=run2)
    p3 = Process(target=run3,args=("进程3",))

    #启动进程
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

    print("--------程序结束！")


if __name__ == '__main__':
    main()