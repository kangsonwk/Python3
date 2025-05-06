import json

from flask import *
from flask_cors import cross_origin

from entity.User import User
from model import userModel

#创建蓝图对象
userBP=Blueprint("userBP",__name__,url_prefix="/user")

#查询
@userBP.route('/findOne',methods=['GET'])
def findOne():
    id=request.args.get("id")
    user=userModel.findOne(id)
    return user

@userBP.route('/queryList',methods=['GET'])
def queryList():
    age=request.args.get("age")
    type = request.args.get("type")
    userList=userModel.findList(age,type)
    return render_template("index5.html",userList=userList)

@userBP.route('/queryAll',methods=['GET'])
def queryAll():
    userList=userModel.findAll()
    return render_template("index5.html",userList=userList)

#删除
@userBP.route('/deleteById',methods=['GET'])
def deleteById():
    id = request.args.get("id")
    userList=userModel.delete(id)
    return "success"

#新增
@userBP.route('/addUser',methods=['POST'])
def addUser():
    params = request.json
    user=User(params.get("id"), params.get("userName"), params.get("passWord"),params.get("name"), params.get("age"), params.get("hobby"), params.get("type"))
    userModel.add(user)
    return "success"

#编辑
@userBP.route('/updateHobby',methods=['POST'])
def updateHobby():
    params = request.json
    userModel.update(params.get("id"),params.get("hobby"))
    return "success"

#前后端分离（返回json数据）
@userBP.route('/queryAll_2',methods=['GET'])
@cross_origin()  #允许跨域请求
def queryAll_2():
    userList=userModel.findAll()
    userDictList=[]
    for user in userList:
        userDict={"name":user.name,"age":user.age,"hobby":user.hobby,"type":user.type}
        userDictList.append(userDict)
    jsonData=json.dumps(userDictList,ensure_ascii=False)
    return jsonData
