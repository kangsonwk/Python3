from database import db
class User(db.Model):
    def __init__(self,id, userName, passWord,name, age, hobby, type):
        self.id = id
        self.userName = userName
        self.passWord = passWord
        self.name = name
        self.age = age
        self.hobby = hobby
        self.type = type

    #实体类和表映射
    __tablename__="user"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    userName = db.Column("user_name",db.String)
    passWord = db.Column("pass_word",db.String)
    age = db.Column(db.Integer)
    hobby = db.Column(db.String)
    type = db.Column(db.String)
