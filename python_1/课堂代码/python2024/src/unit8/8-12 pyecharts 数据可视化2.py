# pyecharts 数据可视化 - 饼图

import pyecharts
from pyecharts.charts import Bar #柱状图
from pyecharts.charts import Pie #饼图
from pyecharts.charts import Line #折线图
from pyecharts import options as opts  #设置参数

def main():
    cate = ['苹果', '华为', '小米', 'Oppo', 'Vivo', '三星']
    data = [153, 124, 107, 99, 89, 46]
    dataList=[]
    for i in range(0,len(cate)):
        d=[cate[i],data[i]]
        dataList.append(d)
    #创建饼图对象
    pie=Pie()
    pie.add("单位：万台",dataList)
    # 设置标题
    pie.set_global_opts(title_opts=opts.TitleOpts(title="手机销售情况"))
    # 生成图表
    pie.render("手机销售情况.html")





if __name__ == '__main__':
    main()