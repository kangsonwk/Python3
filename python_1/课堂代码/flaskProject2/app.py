from flask import Flask,request
import json

app = Flask(__name__)

#get登录
@app.route('/getlogin',methods=['GET'])
def getlogin():
    name=request.args.get("name")
    pwd = request.args.get("pwd")
    print("用户名："+name)
    return '登陆成功！'+name

#post登录
@app.route('/postlogin',methods=['POST'])
def postlogin():
    name=request.form["name"]
    pwd = request.form["pwd"]
    print("用户名："+name)
    return '登陆成功！'+name

#post登录-json (推荐)
@app.route('/jsonlogin',methods=['POST'])
def jsonlogin():
    params=request.json
    name=params.get("name")
    pwd=params.get("pwd")
    print("用户名："+name)
    return '登陆成功！'+name


if __name__ == '__main__':
    app.run()
