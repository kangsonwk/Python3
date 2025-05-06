# 13-4 自定义异常

#声明自定义异常类
class MyException(Exception):
	pass

def fun():
	sex=input("请输入性别：")
	if sex=="男" or sex=="女":
		print("您的性别是：",sex)
	else:
		print("您输入的性别有误！")
		ex=MyException("性别只能是男或者女！") #创建一个异常对象
		raise ex #手动抛出异常


def main():
	try:
		fun()
	except MyException as e:
		print("参数错误......")


#---------------------------
if __name__ == '__main__':
	main()