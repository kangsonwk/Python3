from database import *
from flask import *

from entity.User import User

#根据id查询
def findOne(id):
    user=User.query.filter_by(id=id).first()
    return user

#获取多个结果
def findList(age,type):
    userList=User.query.filter_by(age=age,type=type).all()
    return userList

#查询所有
def findAll():
    userList=User.query.all()
    return userList

#删除
def delete(id):
    db.session.query(User).filter(User.id==id).delete()
    db.session.commit() #提交

#新增
def add(user):
    db.session.add(user)
    db.session.commit() #提交

#编辑
def update(id,hobby):
    user=User.query.filter_by(id=id).first()
    user.hobby=hobby #更新对象的属性
    #将一个对象转换成字典，并获取所有属性和值
    for key,value in user.__dict__.items():
        if key != "id": #排除id属性
            setattr(user,key,value)
    db.session.commit()  # 提交




    db.session.add(user)
    db.session.commit() #提交

