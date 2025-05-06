# 9-5 记账本
import json
import datetime

#读数据（查询）
def read():
    with open(r"data.txt","r")as f:
        jsonData=f.read()
    dataList=json.loads(jsonData)
    return dataList

#写数据（更新）
def write(dataList):
    jsonData=json.dumps(dataList,ensure_ascii=False)
    with open(r"data.txt","w")as f:
        f.write(jsonData)

#显示账单
def showData():
    dataList=read()
    sumIn=0 #总收入
    sumOut = 0  # 总支出
    now = datetime.datetime.now() #当前时间
    print("**************************本月账单*******************************")
    for data in dataList:
        #将字符串转换成datetime
        time=datetime.datetime.strptime(data["时间"],"%Y/%m/%d %H:%M:%S")
        if time.year==now.year and time.month==now.month: #判断是否同年同月
            if data["分类"]=="支出":
                sumOut = sumOut + data["金额"]
                print(data["时间"],"    ",data["项目"],"    ",data["金额"]*-1)
            else:
                sumIn=sumIn+data["金额"]
                print(data["时间"],"    ",data["项目"],"    ",data["金额"])
    print("***********************************************************")
    print("--总收入：",sumIn,"元 ，总支出：",sumOut,"元，结余：",sumIn-sumOut,"元！--")

#新增账单
def addData():
    dataList = read()
    content = input("请输入账单明细：")
    amount = float(input("请输入账单金额："))
    c = int(input("请选择（1.收入 2.支出）："))
    cla="支出"
    if c==1:
        cla = "收入"
    time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    newData={"时间": time, "项目": content, "金额": amount, "分类": cla}
    dataList.append(newData)
    write(dataList)
    print("---------新增成功！")



def main():
    # #写入初始数据
    # d='[{"时间": "2024/01/04 15:20:21", "项目": "彩票中奖", "金额": 1000, "分类": "收入"}]'
    # with open(r"data.txt","w") as f:
    #         f.write(d)
    while True:
        showData()
        choice=input("=====新增账单请输入1：")
        if choice=="1":
            addData()


if __name__ == '__main__':
    main()