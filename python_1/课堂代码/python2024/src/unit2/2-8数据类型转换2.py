# 2-7数据类型转换

#获取数据类型 type()
s=123
# print(type(s))

#int() 转int
s1="365"
ns1=int(s1)
# print(ns1+5)

f1=3.94
nf1=int(f1)
# print(nf1)

#float() 转float
s2="3.14"
ns2=float(s2)
# print(type(ns2))

i2=6
ni2=float(i2)
# print(ni2)

# 字符串转数值类型时，必须是对应数值形态
s3="3.14"
# print(float(s3))

#str() 转字符串
s1=123
f1=3.14

ns1=str(s1)
nf1=str(f1)
# print(type(nf1))

#bool() 转布尔   表示空意义的数据转成False,其他转成True
i1=1
i2=0
f1=1.5
f2=0.0
s1="a"
s2=""
print(bool(s2))




