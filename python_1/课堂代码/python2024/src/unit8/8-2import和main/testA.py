def mySum(lista):
    print("开始求和....")
    return sum(lista)

def myMax(lista):
    print("开始求最大值...")
    return max(lista)

def myMin(lista):
    print("开始求最小值....")
    return min(lista)

#---------外部引入时不会执行
def main(): #入口方法
    print("testA开始执行.....")

if __name__ == '__main__':
    main()
