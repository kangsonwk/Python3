# pyecharts 数据可视化 其他：matplotlib
# *******
# pyecharts 是一个用于生成 Echarts 图表的类库。
# Echarts 是百度开源的一个数据可视化 JS 库。
# 用 Echarts 生成的图可视化效果非常棒.
# 安装 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyecharts


import pyecharts
from pyecharts.charts import Bar #柱状图
from pyecharts.charts import Pie #饼图
from pyecharts.charts import Line #折线图
from pyecharts import options as opts  #设置参数

def main():
    countA={"衬衫":90,"毛衣":12,"裙子":34,"风衣":67,"T恤":99} #A商家销量
    countB = {"衬衫": 27, "毛衣": 66, "裙子": 53, "风衣": 12, "T恤": 70} #B商家销量
    countC = {"衬衫": 50, "毛衣": 77, "裙子": 64, "风衣": 67, "T恤": 22} #C商家销量
    print()

    #创建柱状图对象
    bar=Bar()
    # 添加x轴数据
    bar.add_xaxis(list(countA.keys()))
    #添加y数据
    bar.add_yaxis("A商家销量",list(countA.values()))
    bar.add_yaxis("B商家销量", list(countB.values()))
    bar.add_yaxis("C商家销量", list(countC.values()))
    #设置标题
    bar.set_global_opts(title_opts=opts.TitleOpts(title="商场销售情况"))
    #生成图表
    bar.render("商场销售情况.html")



if __name__ == '__main__':
    main()