# pygame游戏开发
import time
import pygame
import sys
from pygame.locals import * #检测事件

def main():
    #创建窗口
    screen=pygame.display.set_mode((400,300))

    #读取素材
    img1=pygame.image.load("background.jpg") #背景
    img2=pygame.image.load("player.png") #玩家

    a=10 #x轴
    b = 100  # y轴
    direction="right"

    while True:
        #在窗口中加入对象
        screen.blit(img1,(0,0))
        screen.blit(img2, (a, b)) #blit(对象, 位置)

        #刷新窗口,显示所有对象
        pygame.display.update()

        #判断方向
        if a<=0:
            direction = "right"
        if a>=300:
            direction = "left"

        #移动
        if direction=="right":
            a=a+10  #向右移动10个像素
        else:
            a=a-10 #向左移动10个像素
        time.sleep(0.1)

        #检测用户退出
        for event in pygame.event.get(): # 获取所有的用户事件
            if event.type==QUIT:
                print("正在退出.....")
                sys.exit()



if __name__ == '__main__':
    main()