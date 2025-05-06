# 14.7 用户登录和注册
import random

import pymysql

#获取数据库连接公共方法
def getConnection():
    host = "localhost"
    port = 3306  # 记得使用数字
    user = "root"
    password = "123456"
    db = "py2024"  # 数据库名称
    charset = "utf8"

    # 创建数据库连接对象，建立连接
    conn = pymysql.Connect(host=host, port=port, user=user, password=password, db=db, charset=charset)
    # 创建游标对象（1.执行sql语句；2.处理查询结果）
    cursor = conn.cursor()
    return conn,cursor

#查询数据
def getData(sql,args):
    conn,cursor = getConnection()
    data=()
    try:
        print(sql)
        #执行sql
        if args!=None and len(args)!=0:
            cursor.execute(sql,args)
        else:
            cursor.execute(sql)
        #获取多行数据
        data=cursor.fetchall()
    except Exception as e:
        print("出现异常：",e)
    else:
        print("----success")
    finally:
        cursor.close() #关闭游标
        conn.close() #关闭数据库连接
    return data

#更新数据
def updateData(sql,args):
    conn,cursor = getConnection()
    try:
        print(sql)
        # 执行sql
        if args != None and len(args) != 0:
            cursor.execute(sql, args)
        else:
            cursor.execute(sql)
        # 提交数据
        conn.commit()
    except Exception as e:
        print("出现异常：",e)
        return "error"
    else:
        print("----success")
        return "success"
    finally:
        cursor.close() #关闭游标
        conn.close() #关闭数据库连接

#登录
def login():
    uname=input("请输入用户名：")
    upwd=input("请输入密码：")
    sql="SELECT * FROM users WHERE name=(%s) and password=(%s)"
    users=getData(sql, [uname,upwd])
    if len(users)>0:
        print("-----------登录成功！")
        return "success"
    else:
        print("-----------登录失败！")
        return "faild"

#注册
def reg():
    uname = input("请输入用户名：")
    upwd = input("请输入密码：")
    sql = "SELECT * FROM users WHERE name=(%s)"
    users = getData(sql, [uname])
    if len(users)>0:
        print("-----------用户名已存在！")
        return "faild"
    #用户新增
    sql2 = "INSERT INTO users(name,password) VALUES((%s),(%s))"
    updateData(sql2, [uname,upwd])
    print("-------------注册成功！")

def main():
    reg()


#---------------------------
if __name__ == '__main__':
	main()