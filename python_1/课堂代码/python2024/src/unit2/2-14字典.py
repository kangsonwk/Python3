# 2-14字典
# 键:值
#特性：无序；键值对形式；键不可以重复，值可以重复
# 使用字符串作为键，使用任意类型作为值
dicta={"name":"张三","age":18,"hobby":"打球"}

#使用键获取值
print(dicta["name"])
print(dicta["age"])
print(dicta["hobby"])

#修改字典的值
dicta["hobby"]="看书"
print(dicta["hobby"])

#增加数据 -- 给一个原本不存在的键赋值
dicta["sex"]="男"
print(dicta)

#删除数据
dicta.pop("hobby")
print(dicta)

#判断是否存在某个键
print("name" in dicta)








