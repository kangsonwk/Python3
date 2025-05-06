# 9-2 json和python数据转换
import json

def main():
    # 保存一个学员信息
    stu1 = '{"name":"zhangsan","age":18,"hobby":"play"}'

    # 保存多个学员信息
    stus ='[{"name":"zhangsan","age":18,"hobby":"play"},\
           {"name":"lisi","age":18,"hobby":"play"},\
           {"name":"wangqu","age":18,"hobby":"play"}]'

    #1.json转python
    pythonData=json.loads(stu1)
    # print(pythonData["hobby"])
    pythonData2 = json.loads(stus)
    # for stu in pythonData2:
    #     print(stu["name"])

    #2.python转json --- 字典、列表嵌套字典
    pthonStu = {"name":"张三","age":18,"hobby":"play"}
    jsonData=json.dumps(pthonStu,ensure_ascii=False) #禁止ascii转换
    print(jsonData,type(jsonData))

if __name__ == '__main__':
    main()