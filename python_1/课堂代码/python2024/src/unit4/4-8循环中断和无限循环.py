# 4-8循环中断和无限循环

# for i in range(1,11):
#     if i==5:
#         print("第5年，申请了提前还款，今年不用还了！")
#         continue #结束当前循环
#     if i==6:
#         print("第", i, "年，还款24万元！")
#         continue
#     if i==8:
#         print("第", i, "年，提前还清所有贷款！")
#         break #终止整个循环
#     print("第",i,"年，还款12万元！")


#无限循环
i=1
while True:
    print("第",i,"次打印：你好，世界！")
    i=i+1