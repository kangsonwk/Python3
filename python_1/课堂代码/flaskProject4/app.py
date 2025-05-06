from flask import *
from blueprint.userBp import userBP

app = Flask(__name__)

#注册蓝图
app.register_blueprint(userBP)

if __name__ == '__main__':
    app.run()
