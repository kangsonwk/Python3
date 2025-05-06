from flask import Flask

#创建Flask程序对象（实例）
#__name__ 内置变量，表示当前程序或者文件
app = Flask(__name__)


@app.route('/world') #路由
def hello_world():  # put application's code here
    return '你好，世界!'

@app.route('/shanghai') #路由
def hello_shanghai():  # put application's code here
    return '你好，上海!'


if __name__ == '__main__':
    app.run() #启动web程序
