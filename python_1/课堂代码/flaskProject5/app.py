from flask import *
from blueprint.userBp import userBP
from database import app_config, db

#创建web服务
app = Flask(__name__)
app.config.update(app_config)  # 更新应用数据库配置
app.config['JSON_AS_ASCII'] = False #解决中文乱码的问题，将json数据内的中文正常显示
db.init_app(app)  # 初始化 SQLAlchemy，并将 app 作为参数传入

#注册蓝图
app.register_blueprint(userBP)

if __name__ == '__main__':
    app.run()
