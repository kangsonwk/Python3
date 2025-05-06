# 14.6 更新 删除
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
        #更新
        sql1= "UPDATE stu set age=13 WHERE age=134"

        #删除
        sql2 = "DELETE FROM stu WHERE hobby='游戏'"
        print(sql2)
        #执行sql
        cursor.execute(sql2)
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