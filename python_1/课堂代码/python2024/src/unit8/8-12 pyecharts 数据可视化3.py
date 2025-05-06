# pyecharts 数据可视化 - 折线图

import pyecharts
from pyecharts.charts import Bar #柱状图
from pyecharts.charts import Pie #饼图
from pyecharts.charts import Line #折线图
from pyecharts import options as opts  #设置参数

def main():
    x=["1月","2月","3月","4月","5月","6月"]
    shenzhen=[100,200,300,200,100,400]
    changsha=[50,100,200,300,400,100]
    line=Line()
    line.add_xaxis(x)
    line.add_yaxis("深圳",shenzhen)
    line.add_yaxis("长沙", changsha)
    # 设置标题
    line.set_global_opts(title_opts=opts.TitleOpts(title="降雨量图"))
    # 生成图表
    line.render("降雨量图.html")






if __name__ == '__main__':
    main()