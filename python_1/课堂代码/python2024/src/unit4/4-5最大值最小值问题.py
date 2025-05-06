# 4-5最大值最小值问题
scores=[67,78,68,87,92,45,69,77]

#最高分和最低分
# maxScore=scores[0]
# minScore=scores[0]
# for score in scores:
#     if score>maxScore:
#         maxScore=score #将更大的当前值更新为最大值
#     if score<minScore:
#         minScore=score #将更小的当前值更新为最小值
# print(maxScore,minScore)

#获取最长字符串
strings=["hello","world","你好世界","你是谁？","who are you?"]
longest=len(strings[0])
result=""
for s in strings:
    l=len(s)
    if l>longest:
        longest=l
        result=s
print("最长字符串:",result,"长度：",longest)

