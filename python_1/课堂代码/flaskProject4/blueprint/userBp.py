from flask import *
from entity.User import User

#创建蓝图对象
userBP=Blueprint("userBP",__name__,url_prefix="/user")

#接口

#手动加标签
@userBP.route('/model1',methods=['GET'])
def model1():
    return '<h1>你好，世界！<h1>'

#返回已有的html
@userBP.route('/model2',methods=['GET'])
def model2():
    return render_template("index1.html")

#将参数传递到页面
@userBP.route('/model3',methods=['GET'])
def model3():
    name="张三"
    age=18
    return render_template("index2.html",name=name,age=age)

#将对象传递到页面
@userBP.route('/model4',methods=['GET'])
def model4():
    name=request.args.get("name")
    age=request.args.get("age")
    hobby=request.args.get("hobby")
    type=request.args.get("type")
    user=User(name,age,hobby,type)
    return render_template("index3.html",user=user)

#模版-选择结构
@userBP.route('/model5',methods=['GET'])
def model5():
    name=request.args.get("name")
    age=request.args.get("age")
    hobby=request.args.get("hobby")
    type=request.args.get("type") #用户类型：admin---管理员  其他--普通用户
    user=User(name,age,hobby,type)
    return render_template("index4.html",user=user)

#模版-循环
@userBP.route('/model6',methods=['GET'])
def model6():
    user1 = User("张三", 18, "打球", "admin")
    user2 = User("李四", 17, "看电影", "user")
    user3 = User("王五", 19, "音乐", "admin")
    user4 = User("赵六", 16, "看书", "admin")
    userList = [user1, user2, user3, user4]
    return render_template("index5.html",userList=userList)