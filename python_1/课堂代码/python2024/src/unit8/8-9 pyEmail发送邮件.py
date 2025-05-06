# pyEmail 发送邮件
# 安装 pip install pyEmail

import smtplib   #smtp协议包
from email.mime.text import MIMEText #用于构建邮箱内容

def main():
    msg_from = "test666@126.com"  # 发件人
    password="JLCHZNCJCHDLPKMY" #客户端授权码
    msg_to = "2268902957@qq.com"  # 收件人

    #构建邮箱内容
    subject="测试邮件0127" #主题
    content="服务器出现异常，请速回！" #正文
    msg=MIMEText(content) #msg邮箱内容对象
    msg["Subject"]=subject
    msg["From"] = msg_from
    msg["To"] = msg_to

    #发送邮件
    smtpObj=smtplib.SMTP_SSL("smtp.126.com",465)
    smtpObj.login(msg_from,password) #登录
    smtpObj.sendmail(msg_from,msg_to,str(msg)) #发送邮件
    print("发送成功！")
    smtpObj.quit()




if __name__ == '__main__':
    main()