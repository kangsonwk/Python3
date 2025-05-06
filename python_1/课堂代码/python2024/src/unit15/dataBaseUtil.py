#数据库操作公共方法
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
    finally:
        cursor.close() #关闭游标
        conn.close() #关闭数据库连接
    return data

#更新数据
def updateData(sql,args):
    r=0 #保存更新成功的条数
    conn,cursor = getConnection()
    try:
        print(sql)
        # 执行sql
        if args != None and len(args) != 0:
            r=cursor.execute(sql, args)
        else:
            r=cursor.execute(sql)
        # 提交数据
        conn.commit()
    except Exception as e:
        print("出现异常：",e)
    finally:
        cursor.close() #关闭游标
        conn.close() #关闭数据库连接
    return r