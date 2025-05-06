import time
import pygame
from pygame.locals import *   #检测事件，如监控键盘按键
import sys
import random


#玩家类：
#属性：显示窗口、位置、图片、子弹列表、移动状态
#方法：显示、移动、开火
class Player():
	def __init__(self,screen):
		self.screen=screen #将一个窗口对象作为属性值，表示玩家对象现实的窗口
		self.x=150
		self.y=500
		self.img=pygame.image.load("feiji/hero1.png") #图片
		self.bulletList=[]#子弹列表
		self.moveLeftState=0 #1移动 0不移动
		self.moveRightState = 0  # 1移动 0不移动
	#显示
	def display(self):
		#显示玩家
		self.screen.blit(self.img,(self.x,self.y))
		#显示玩家所有子弹
		for bullet in self.bulletList:
			bullet.display()
			bullet.move()
			if bullet.y<=0: #清除不需要的子弹对象
				self.bulletList.remove(bullet)
	#移动
	def move(self):
		if self.moveLeftState==1 and self.x>=-30:
			self.x=self.x-5
		elif self.moveRightState==1 and self.x<=270:
			self.x = self.x+5

	#开火
	def fire(self):
		bullet=PlayerBullet(self.screen,self.x,self.y)
		self.bulletList.append(bullet)


#玩家子弹类：
#属性：显示窗口、位置、图片
#方法：显示、移动
class PlayerBullet():
	def __init__(self, screen,x,y):
		self.screen = screen  # 将一个窗口对象作为属性值，表示玩家对象现实的窗口
		self.x = x+40
		self.y = y-20
		self.img = pygame.image.load("feiji/bullet.png")  # 图片

	# 显示
	def display(self):
		# 显示子弹
		self.screen.blit(self.img, (self.x, self.y))

	# 移动
	def move(self):
		self.y=self.y-20 #向上移动20个像素

#敌机类
#属性：显示窗口、位置、图片、子弹列表、移动状态
#方法：显示、移动、开火
class Enemy():
	def __init__(self, screen):
		self.screen = screen  # 将一个窗口对象作为属性值，表示玩家对象现实的窗口
		self.x = 0
		self.y = 0
		self.img = pygame.image.load("feiji/enemy0.png")  # 图片
		self.bulletList = []  # 子弹列表
		self.moveState = 1  #1向左移动 2向右移动

	# 显示
	def display(self):
		# 显示敌机
		self.screen.blit(self.img, (self.x, self.y))
		# 显示敌机所有子弹
		for bullet in self.bulletList:
			bullet.display()
			bullet.move()
			if bullet.y>=600: #清除不需要的子弹对象
				self.bulletList.remove(bullet)

	# 移动
	def move(self):
		if self.moveState==1:
			self.x=self.x-5 #向左移动五个像素
		elif self.moveState==2:
			self.x = self.x + 5  # 向左移动五个像素
		#当移动到边缘时改变方向
		if self.x<=20:
			self.moveState=2
		if self.x>280:
			self.moveState = 1
	# 开火
	def fire(self):
		bullet = EnemyBullet(self.screen,self.x,self.y)
		self.bulletList.append(bullet)
		print("敌机开火！")

#敌机子弹类
#属性：显示窗口、位置、图片
#方法：显示、移动
class EnemyBullet():
	def __init__(self, screen, x, y):
		self.screen = screen  # 将一个窗口对象作为属性值，表示玩家对象现实的窗口
		self.x = x+20
		self.y = y+30
		self.img = pygame.image.load("feiji/bullet2.png")  # 图片
	# 显示
	def display(self):
		# 显示子弹
		self.screen.blit(self.img, (self.x, self.y))
	# 移动
	def move(self):
		self.y=self.y+20


#捕捉用户操作:左右移动、开火、退出
def key_control(player):
	for event in pygame.event.get():
		if event.type==QUIT:
			print("正在退出.....")
			sys.exit(0) #强制退出
		elif event.type==KEYDOWN: #键盘按下
			if event.key==K_SPACE: #捕捉空格键
				player.fire()
				print("玩家开火！")
			if event.key==K_LEFT: #捕捉左键
				player.moveLeftState=1
				print("玩家向左！")
			if event.key==K_RIGHT: #捕捉右键
				player.moveRightState=1
				print("玩家向右！")
		elif event.type == KEYUP:  # 键盘抬起
			if event.key == K_LEFT:  # 捕捉左键
				player.moveLeftState = 0
				print("停止向左！")
			if event.key == K_RIGHT:  # 捕捉右键
				player.moveRightState = 0
				print("停止向右！")

#mian方法
def main():
	# 创建窗口对象
	screen=pygame.display.set_mode((300,600))
	#读取背景图片
	background = pygame.image.load("feiji/background.png")
	#创建玩家对象，并传入窗口
	player=Player(screen)
	# 创建敌机对象，并传入窗口
	enemy = Enemy(screen)
	#循环显示所有对象并刷新
	while True:
		screen.blit(background,(0,0)) #显示背景
		player.display()#显示玩家
		enemy.display()  # 显示敌机
		player.move() #玩家移动
		enemy.move()  # 玩家移动
		n=random.randint(1,10)
		if n==1:
			enemy.fire() #敌机随机开火
		key_control(player) #捕捉用户操作
		pygame.display.update() #刷新窗口
		time.sleep(0.05)


#---------------------------
if __name__ == '__main__':
	main()