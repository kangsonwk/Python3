# 14-3 连接mysql
import pymysql

class main():
    host="localhost"
    port=3306 #记得使用数字
    user="root"
    password="123456"
    db="py2024" #数据库名称
    charset="utf8"

    #创建数据库连接对象，建立连接
    conn=pymysql.Connect(host=host,port=port,user=user,password=password,db=db,charset=charset)
    try:
        print("数据库已连接.....")
        #......相关操作
    except Exception as e:
        print("出现异常：",e)
    finally:
        conn.close() #关闭数据库连接


#---------------------------
if __name__ == '__main__':
	main()