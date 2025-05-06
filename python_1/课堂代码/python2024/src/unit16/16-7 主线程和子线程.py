# 16-7 主线程和子线程
import time
#程序执行时，会有一个默认进程（主进程），进程中也包含一个默认线程（主线程）
#手动创建的线程对象，称为子线程（进程同理）
#主线程并不会等待子线程执行完毕，会直接执行后面的内容（进程同理）



from threading import Thread
def run(name):
    print(name+"执行了任务")
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

    #需要将当前线程执行完毕，才能执行后面内容
    t1.join() #将子线程加入到主线程
    t2.join()
    t2.join()

    print("--------程序结束！")


if __name__ == '__main__':
    main()