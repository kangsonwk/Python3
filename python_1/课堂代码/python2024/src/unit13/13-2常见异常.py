# 13-2 常见异常


def main():
    # SyntaxError 语法错误
    # 中英文符号 ：:   缺少某些符号: "  )

    # NameError: name 'b' is not defined  变量名写错
    # 拼写  大小写 Il

    # IndentationError  缩进错误
    # 缺少或有多余缩进
    # 缩进不一致
    # if 1==1:
    # 	print(1)
    #     print(1)

    # ModuleNotFoundError  第三方包未安装
    # 注意模块名称和安装方法

    # FileNotFoundError 文件未找到

    # AttributeError: 'int' object has no attribute 'eat'  对象没有某个属性和方法
    # 检查名称 、 是否私有化

    # IndexError: list index out of range 索引越界
    # listA=[131,3,22,43,5,46,6]
    # print(listA[7])

    # TypeError: can only concatenate str (not "int") to str 类型错误

    # KeyError: 'c' 键错误
    # dict1={"name":"zhangsan","age":12}
    # print(dict1["hobby"])

    # ImportError：导入模块失败。
    # ModuleNotFoundError：模块未找到（ImportError的子类，Python3.6+）。
    # StopIteration：迭代器没有更多元素。
    # AssertionError：assert语句条件为假。
    # ZeroDivisionError: division by zero 除数错误

    # 操作系统相关异常（OSError子类）
    # 	PermissionError：无操作权限。

    # 	FileExistsError：创建已存在的文件/目录。

    # 	TimeoutError：操作超时。

    # 其他常见异常
    # 	RuntimeError：通用运行时错误（如递归深度过大）。

    # 	NotImplementedError：抽象方法未实现。

    # 	MemoryError：内存不足。

    # 	OverflowError：数值运算超出范围（如整数过大）。

    # 	UnicodeError：编码/解码错误（如UnicodeEncodeError）。

    # 特殊异常（继承自BaseException）
    # 	KeyboardInterrupt：用户按下中断键（如Ctrl+C）。

    # 	SystemExit：sys.exit()触发，程序终止。

    a = 5
    b = 0
    print(a / b)


# ---------------------------
if __name__ == '__main__':
    main()
