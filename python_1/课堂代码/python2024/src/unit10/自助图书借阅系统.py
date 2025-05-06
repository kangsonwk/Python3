#一个json文档只能保存一种业务数据
import json
import datetime

# #用户信息
# d1='[{"用户名": "zhangsan", "密码": "123", "姓名": "张三", "类型": "user"},{"用户名": "aaa", "密码": "123", "姓名": "李四", "类型": "admin"}]'
# with open(r"users.txt","w") as f:
# 	f.write(d1)
#
# #图书信息
# d2='[{"编号":1001, "书名": "红楼梦", "作者": "曹雪芹", "借出状态": "可借"},\
# {"编号":1002, "书名": "java教程","作者": "齐一天", "借出状态": "可借"},\
# {"编号":1003, "书名": "python开发指南","作者": "耶稣", "借出状态": "已借出"},\
# {"编号":1004, "书名": "李白诗集","作者": "李白", "借出状态": "可借"}\
# ]'
# with open(r"books.txt","w") as f:
# 	f.write(d2)
#
#借出明细
d3='[{"书名": "python开发指南", "编号":1003,"借书人": "zhangsan", "借阅时间": "2023/01/01 15:20:21","应还时间": "2023/01/04 15:20:21","归还状态": "未归还"}]'
with open(r"borrowing.txt","w") as f:
	f.write(d3)

#用户信息
userInfo={}

#读数据（查询）
def read(fileName):
    with open(fileName,"r")as f:
        jsonData=f.read()
    dataList=json.loads(jsonData)
    return dataList

#写数据（更新）
def write(dataList,fileName):
    jsonData=json.dumps(dataList,ensure_ascii=False)
    with open(fileName,"w")as f:
        f.write(jsonData)

# 1.用户登录；login()
def login():
	global userInfo
	msg="失败"
	usersList=read("users.txt")
	name=input("请输入用户名：")
	pwd=input("请输入密码：")
	for user in usersList:
		if name==user["用户名"] and pwd==user["密码"]:
			userInfo=user
			print("---恭喜你登陆成功！",user["姓名"],"！")
			msg="成功"
			break
	if msg=="失败":
		print("-----验证失败！")
	return msg


# 2.显示图书列表；showAllBooks()
def showAllBooks():
	booksList=read("books.txt")
	print("--编号----书名----作者----借出状态--")
	for book in booksList:
		print(book["编号"],"  ",book["书名"],"  ",book["作者"],"  ",book["借出状态"])
	print("---------------------------------")

# 3.图书上架；addBook()
def addBook():
	if userInfo["类型"]!="admin":
		print("没有权限！")
		return
	bookList=read("books.txt")
	numList=[]
	for book in bookList:
		numList.append(book["编号"])
	num=max(numList)+1
	name=input("请输入书名：")
	author=input("请输入作者：")
	state="可借"
	newBook={"编号":num, "书名": name, "作者": author, "借出状态": state}
	bookList.append(newBook)
	write(bookList,"books.txt")
	print("新书",name,"已上架！")

# 4.图书下架；delBook()
def delBook():
	if userInfo["类型"]!="admin":
		print("没有权限！")
		return
	bookList = read("books.txt")
	num=int(input("请输入要下架的图书编号："))
	exist=0  #0不存在 1存在
	for book in bookList:
		if num==book["编号"]:
			exist=1
			if book["借出状态"]=="已借出":
				print("图书", book["书名"], "已借出！")
				break
			bookList.remove(book)
			write(bookList, "books.txt")
			print("图书",book["书名"],"已下架！")
			break
	if exist==0:
		print("您输入的图书信息不存在！")

# 5.借书；lendBook()
def lendBook():
	time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") #当前时间
	bookList = read("books.txt")
	num = int(input("请输入要借阅的图书编号："))
	day = int(input("请输入要借阅的时长（天）："))
	exist = 0  # 0不存在 1存在
	for book in bookList:
		if num==book["编号"]:
			exist = 1
			if book["借出状态"]=="可借":
				print("您已借出图书：",book["书名"], "！")
				#更新图书状态
				book["借出状态"]="已借出"
				write(bookList, "books.txt")
				state="未归还"
				#添加借书明细
				borrowList = read("borrowing.txt")
				backTime=datetime.datetime.now()+datetime.timedelta(days=day)
				backTime = backTime.strftime("%Y/%m/%d %H:%M:%S")  # 应还时间
				b={"书名": book["书名"], "编号": book["编号"],"借书人":userInfo["用户名"], "借阅时间":time,\
				 "应还时间": backTime, "归还状态":state}
				borrowList.append(b)
				write(borrowList, "borrowing.txt")
				break
			else:
				print(book["书名"],"已借出！下次再来吧！")
				break
	if exist == 0:
		print("您输入的图书信息不存在！")

# 6.查询图书；findBook()
def findBook():
	booksList = read("books.txt")
	exist = 0  # 0不存在 1存在
	name=input("请输入要查询的书名：")
	print("-----------------------------------------")
	for book in booksList:
		if name==book["书名"]:
			exist = 1
			print(book["编号"], "  ", book["书名"], "  ", book["作者"], "  ", book["借出状态"])
	if exist == 0:
		print("您查询的图书信息不存在！")
	print("-----------------------------------------")

# 7.还书；returnBook()
def returnBook():
	bookList = read("books.txt")
	borrowList = read("borrowing.txt")
	num = int(input("请输入要归还的图书编号："))
	#验证
	exist1 = 0  # 0不存在 1存在
	for book in bookList:
		if num==book["编号"]:
			exist1=1
			if book["借出状态"]=="可借":
				print("已经是归还状态，不要重复归还！")
				return
		else:
			pass
	if exist1 == 0:
		print("您查询的图书信息不存在！")
		return
	exist2 = 0  # 0不存在 1存在
	for borrow in borrowList:
		if num==borrow["编号"] and userInfo["用户名"]==borrow["借书人"]:
			exist2=1
			#更新借阅明细
			borrow["归还状态"]="已归还"
			write(borrowList, "borrowing.txt")
			# 更新图书状态
			for book in bookList:
				if num == book["编号"]:
					book["借出状态"]="可借"
					write(bookList, "books.txt")
					print("图书",book["书名"],"已归还！")
			break
	if exist2 == 0:
		print("借书明细不存在！")
		return


# 8.用户借书明细: lendInfo()
def lendInfo():
	time = datetime.datetime.now()
	borrowList = read("borrowing.txt")
	print("--编号----书名----借阅时间----应还时间----归还状态----备注--")
	for borrow in borrowList:
		if userInfo["用户名"]==borrow["借书人"]:
			lendTime=datetime.datetime.strptime(borrow["应还时间"],"%Y/%m/%d %H:%M:%S")
			t=lendTime-time #时间相减
			hour=t.seconds/3600 #获取小时数
			msg=""
			if hour<0:
				msg="已逾期"
			elif hour>=0 and hour<=24:
				msg="不足一天，尽快归还"
			print(borrow["编号"], "  ", borrow["书名"], "  ", borrow["借阅时间"], "  ", borrow["应还时间"], "  ", borrow["归还状态"], "  ",msg)
	print("---------------------------------")





#-------------------------------------------------------------------------
def main():
	print("**************************自助图书借阅系统 1.0*****************************")
	result=login()
	# result = "成功"
	if result=="成功":
		while True:
			print("1.显示所有图书；\n2.上架；\n 3.下架；\n4.借书；\n5.还书；\n6.查询图书；\n7.查看借阅信息；\n8.退出；")
			print("*******************************************")
			c=int(input("请输入功能编号："))
			if c==1:
				showAllBooks()
			elif c==2:
				addBook()
				showAllBooks()
			elif c == 3:
				delBook()
				showAllBooks()
			elif c == 4:
				lendBook()
			elif c == 5:
				returnBook()
			elif c == 6:
				findBook()
			elif c == 7:
				lendInfo()
			elif c == 7:
				break
			else:
				print("没有此业务！")




if __name__ == '__main__':
    main()