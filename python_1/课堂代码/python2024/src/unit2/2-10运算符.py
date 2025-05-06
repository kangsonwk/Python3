# 2-10运算符


#算术运算符：数学运算中的符号
# +-*/  ...
a=10
b=3
# print(a/b) #除法
# print(a//b) #除法，只保留整数部分
# print(a%b) #获取余数 （模）
# print(a**b) #获取幂次方


#赋值运算符：给变量赋值
# = += -+  ...
a=125
b=a+1
a+=5 #在a的基础上+5   ---- a=a+5
a-=5
a*=5
# print(a)


#关系运算符：比较两个变量之间关系
#> <  ==  >=  <=  !=
#关系运算符得到的运算结果是布尔值
a=5
b=6
# print(a==b)
# print(a>b)
# print(a<b)
# print(a>=b)
# print(a<=b)
# print(a!=b)

#比较字符串是否相等
c="hello"
d="hello"
# print(c==d)


#逻辑运算符：判断表达式之间的逻辑关系
# and or  not
print(5>3 and 6<3)
print(5>3 or 6<3)
print(not 6<3)
