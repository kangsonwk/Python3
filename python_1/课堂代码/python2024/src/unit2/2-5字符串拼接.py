# 2-5 字符串拼接

# +拼接
a1="张三"
a2="李四"
a3="王五"
c=a1+"爱"+a2
# print(c)
# print(a1+a2+a3) #做了拼接
# print(a1,a2,a3) #分别打印了多个内容，没有拼接

#format函数拼接
s1="hello!{}".format(a2)
s2="hello!{}{}{}".format(a1,a2,a3)
s3="hello!{0}{1}{2}".format(a1,a2,a3)
s4="hello!{x}{y}{z}".format(x=a1,y=a2,z=a3)

print(s4)