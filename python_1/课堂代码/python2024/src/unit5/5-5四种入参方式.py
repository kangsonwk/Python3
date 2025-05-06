#5-5 四种入参方式

# #参数和返回值案例
# def juicer(f1,f2):
#     print("榨汁机开始工作！")
#     juice=f1+f2+"汁"
#     return juice
# j=juicer("香蕉","葡萄")
# print("榨出一杯：",j)

#位置参数-按顺序传参
def function1(nameA,nameB):
    print(nameA,"欠",nameB,"1000元！")
# function1("张三","李四")


#关键字参数 -- 通过变量名传参
# function1(nameB="李四",nameA="张三")

#默认参数 -- 给参数设置默认值
# def function2(nameB,nameA="张三"):
#     print(nameA,"欠",nameB,"1000元！")
# function2("王五")


#可变长度参数 -- 接收任意个参数
def function3(*names):
    print("张三的债主们：")
    print(names,type(names))
function3("大宝","二宝","李四","王五")


