# 13-3 捕获异常

# try:
# 	可能出现异常的代码
# except Exception as e:
# 	出现异常时执行的内容
# else:
# 	没有异常时执行的内容
# finally:
# 	无论是否异常，都会执行


def fun(listA):
    for i in listA:
        try:
            r = 10 / i
            print(r)
        except TypeError as e:
            print("出现TypeError错误：", e)
        except ZeroDivisionError as e:
            print("出现ZeroDivisionError错误：", e)
        else:
            print("本次循环正常执行！无异常")
        finally:
            print("本次除数：", i)


def main():
    listA = [13, 243, 5, 34, 646, 57, 576, "a", 23, 34, 0, 7878, 7, 989, 898]
    fun(listA)


# ---------------------------
if __name__ == '__main__':
    main()
