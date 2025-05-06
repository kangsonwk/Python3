from flask_sqlalchemy import SQLAlchemy

# 依赖模块： flask_sqlalchemy

# flask-sqlalchemy 在安装/使用过程中,
# 如果出现 ModuleNotFoundError: No module named ('MySQLdb’错误, '
# '则表示缺少mysql依赖包:
# pip install flask_sqlalchemy
# pip install mysqlclient
# pip install pymysql

# 数据库配置
app = None  # 这个变量可以在实际使用时被设置
app_config = {
    'SQLALCHEMY_DATABASE_URI': 'mysql://root:123456@localhost:3306/py2024'
}
# 初始化 SQLAlchemy
db = SQLAlchemy()