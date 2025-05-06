# 14.5 新增数据
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
    return conn

def main():
    conn=getConnection()
    # 创建游标对象（1.执行sql语句；2.处理查询结果）
    cursor = conn.cursor()
    try:
        print("数据库已连接.....")
        for i in range(100):
            name=random.choice(["大宝","二宝","小宝","赵六"])
            age=random.choice([11,12,134,15])
            hobby=random.choice(["看书","打球","游戏","音乐"])

            #sql1= "INSERT INTO stu(name,age,hobby) VALUES('"+name+"',"+str(age)+",'"+hobby+"')"
            sql2 = "INSERT INTO stu(name,age,hobby) VALUES((%s),(%s),(%s))"
            print(sql2)
            #执行sql
            cursor.execute(sql2,[name,age,hobby])
            #提交数据
            conn.commit()
    except Exception as e:
        print("出现异常：",e)
    finally:
        cursor.close() #关闭游标
        conn.close() #关闭数据库连接


#---------------------------
if __name__ == '__main__':
	main()