from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *

from AGENT import AGENT
from Maze import Maze
from Falcon import FALCON
# from main import MNT

from Ui_TD_Falcon import Ui_Form  #导入创建的GUI类

import os
import sys
from math import sin,cos,pi
#自己建一个mywindows类，mywindow是自己的类名。QtWidgets.QMainWindow：继承该类方法
class mywindow(QtWidgets.QMainWindow,Ui_Form):

    #__init__:析构函数，也就是类被创建后就会预先加载的项目。
    # 马上运行，这个方法可以用来对你的对象做一些你希望的初始化

    MAXSTEP = 500
    Icon_x = 31.25
    Icon_y = 28.125
    graphic = True


    __target = []
    __bearing = []
    __currentBearing = []
    __targetBearing = []
    __current = [[]]
    __sonar = []
    __avsonar = []
    __range = []
    __numSonar = 5
    __mines = [[]]
    __path = [[[]]]
    __numStep = []
    __maxStep = 30
    __sonar_mode:bool



    def __init__(self, agt_num : int, m):
    #这里需要重载一下mywindow，同时也包含了QtWidgets.QMainWindow的预加载项。
        super(mywindow, self).__init__()
        self.setupUi(self)

        self.__agent_num = agt_num
        self.init_Panel(0, m) #多个智能体面板的时候这里就要修改
        self.doRefresh(m)


    def init_Panel(self, agt:int, m): #这个mode参数暂时没用 mode = true: snoar || false: avsonar

        # Parameter of Maze Panel
        self.__numStep = [0] * self.__agent_num
        self.__current = [([0] * 2) for i in range(self.__agent_num)]
        m.get_all_Current_to_path( self.__current )

        self.__path = [[[0,0] for i in range(self.__agent_num)] for i in range(self.MAXSTEP)]
        self.__target = m.getTarget()

        self.__mines = [([0] * m.size) for i in range(m.size)]  #二维数组
        for i in range(m.size):
            for j in range(m.size):
                self.__mines[i][j] = m.getMines(i,j)


        self.__currentBearing = m.get_all_CurrentBearing()
        self.__bearing = [0] * self.__agent_num
        self.__targetBearing = m.get_all_TargetBearing()
        self.__sonar = [([0] * 5) for i in range(self.__agent_num)]
        self.__avsonar = [([0] * 5) for i in range(self.__agent_num)]
        self.__maxStep = 30


        self.loadImageIcons(m)

        # Parameter of Sonar Panel
        self.Sonar_r = 100 #Sonar输入的显示的圆的最大半径
        self.Bearing_r = 148  #Current Bearing显示的圆的最大半径

        self.__numSonar = 5

        # self.__sonar_mode = mode
        m.getSonar(agt,self.__sonar[agt])
        m.getAVSonar(agt,self.__avsonar[agt])

        self.__range = m.get_all_Range()

        self.draw_mazePanel(m)

#                            Part of Maze Panel
    def loadImageIcons(self, m):
        self.L_background = QtWidgets.QLabel(self.MineField)
        self.L_mines = [QtWidgets.QLabel(self.MineField) for i in range(m.numMines)]
        self.L_target = QtWidgets.QLabel(self.MineField)
        self.L_tank = [QtWidgets.QLabel(self.MineField) for i in range(self.__agent_num)]

                #   Load Minefield
        self.Background = QtGui.QPixmap('./images/background.jpg')
        self.Targeticon = QtGui.QPixmap('./images/target.png')
        self.Mineicon = QtGui.QPixmap('./images/bomb.png')
        self.Tankicon = []
        for i in range(8):
            self.Tankicon.append(QtGui.QPixmap('./images/tank' + str(i) + '.png'))

    def doRefresh(self, m ): #配合doReset

        self.__target = m.getTarget()
        self.setCurrent(m)
        self.setSonar(m )
        self.setBearing(m)
        self.update()
        for i in range(m.size):
            for j in range(m.size):
                self.__mines[i][j] = m.getMines(i,j)

        self.__numStep = [0] * self.__agent_num
        if(self.graphic):
            self.RePaint(m)

    def doRefresh_Step(self, m): #配合doStep
        self.setCurrent(m)
        self.setSonar(m )
        self.setBearing(m)
        self.update()
        for agt in range(self.__agent_num):
            self.__numStep[agt] += 1
        if(self.graphic):
            self.RePaint(m)

    def setCurrent(self, m ):

        m.get_all_Current_to_path( self.__current )


    def setCurrentPath(self, m ,pos : list,step : int):

        for agt in range(self.__agent_num):
            self.__path[step][agt] = pos[agt]
            self.__numStep[agt] = step


    def draw_mazePanel(self, m):

        #    Paint Minefield-Background


        self.L_background.setObjectName('L_background')
        self.L_background.setGeometry(QtCore.QRect(0, 0, 500, 450))
        self.L_background.setPixmap(self.Background)
        self.L_background.setScaledContents(True)


        #   Paint MineField-Mines
        mine_cnt = 0
        for i in range(m.size):
            for j in range(m.size):
                if(self.__mines[i][j] == 1):
                    self.L_mines[mine_cnt].setObjectName('L_mines'+str(mine_cnt))
                    self.L_mines[mine_cnt].setGeometry(QtCore.QRect(i * self.Icon_x, j * self.Icon_y, self.Icon_x, self.Icon_y))
                    self.L_mines[mine_cnt].setPixmap(self.Mineicon)
                    self.L_mines[mine_cnt].setScaledContents(True)
                    mine_cnt += 1
        #   Paint MineField-Target
        self.L_target.setObjectName('L_target')
        self.L_target.setGeometry(QtCore.QRect(self.__target[0] * self.Icon_x, self.__target[1] * self.Icon_y, self.Icon_x, self.Icon_y))
        self.L_target.setPixmap(self.Targeticon)
        self.L_target.setScaledContents(True)

        # Paint Minefield-Tank

        for i in range(self.__agent_num):
            self.L_tank[i].setObjectName('L_tank'+str(i))
            self.L_tank[i].setGeometry(QtCore.QRect(self.__current[i][0] * self.Icon_x, self.__current[i][1] * self.Icon_y,self.Icon_x,self.Icon_y))
            self.L_tank[i].setPixmap(self.Tankicon[self.__currentBearing[i]])
            self.L_tank[i].setScaledContents(True)
        #for the purpose of tracking the actions of agents

        # if(MNT.Track):
        #     #这一部分是画路径的可以逃过
        #     pass


#                            绘画事件
    def paintEvent(self,e):
        qp = QPainter()
        qp.begin(self)
        self.draw_SonarPanel(0,qp)#绘制多个Agt的方法先搁置
        # self.draw_SonarPanel(0,False,qp)
        self.draw_BearingPanel(0,qp)
        qp.end()



#                             Part of Sonar Panel

    def setSonar(self, m ):
        for agt in range(self.__agent_num):
            m.getSonar(agt,self.__sonar[agt])
            m.getAVSonar(agt,self.__avsonar[agt])

    def get_sonar_mode(self):
        return self.__sonar_mode

    def draw_SonarPanel(self,agt : int,qp):

        pen = QPen(QtGui.QColor('green'))
        qp.setPen(pen)
        qp.setBrush(QtGui.QColor('green'))
        qp.setOpacity(1)

        for i in range(self.__numSonar):
            Rect_x  = 550 - 50 * self.__sonar[agt][i] + i * self.Sonar_r
            Rect_y = 100 - 50 * self.__sonar[agt][i]
            Radius = self.Sonar_r * self.__sonar[agt][i]
            qp.drawEllipse(Rect_x, Rect_y, Radius, Radius)

        pen = QPen(QtGui.QColor('yellow'))
        qp.setPen(pen)
        qp.setBrush(QtGui.QColor('yellow'))
        qp.setOpacity(1)
        for i in range(self.__numSonar):
            Rect_x  = 550 - 50 * self.__avsonar[agt][i] + i * self.Sonar_r
            Rect_y = 300 - 50 * self.__avsonar[agt][i]
            Radius = self.Sonar_r * self.__avsonar[agt][i]
            qp.drawEllipse(Rect_x, Rect_y, Radius, Radius)

#                            Part of Bearing Panel

    def setBearing(self, m ):
        self.__currentBearing = m.get_all_CurrentBearing()
        self.__target = m.getTarget()
        self.__targetBearing = m.get_all_TargetBearing()

    def draw_BearingPanel(self,agt : int, qp):
        def get_circle_point(x0 : float, y0 : float, r : float, angle : int):

            x1 = int(x0 + r * cos(angle * pi / 180))

            y1 = int(y0 + r * sin(angle * pi /180))
            return x1, y1

        currentBearing = self.__currentBearing[agt]
        targetBearing = self.__targetBearing[agt]
        Rect_x = 554.5
        Rect_y = 442
        Radius = self.Bearing_r
        pen = QPen(QtGui.QColor('Blue'))
        qp.setPen(pen)
        qp.setBrush(QtGui.QColor('Blue'))
        qp.setOpacity(1)
        qp.drawEllipse(Rect_x, Rect_y, Radius, Radius)
        qp.drawEllipse(Rect_x + 243, Rect_y, Radius, Radius)
        pen = QPen(QtGui.QColor('Red'), 3, Qt.SolidLine)
        qp.setPen(pen)
        Line_x0 = 628.5
        Line_y0 = 516
        Line_x1 ,Line_y1 = get_circle_point(Line_x0, Line_y0, Radius /2,45 * currentBearing - 90)
        qp.drawLine(Line_x0, Line_y0, Line_x1, Line_y1)
        Line_x0 = 871.5
        Line_y0 = 516
        Line_x1 ,Line_y1 = get_circle_point(Line_x0, Line_y0, Radius /2 ,45 * targetBearing - 90)
        qp.drawLine(Line_x0, Line_y0, Line_x1, Line_y1)




#                           重绘事件
    def RePaint(self, m):
        # Paint Minefield-Mines

        mine_cnt = 0
        for i in range(m.size):
            for j in range(m.size):
                if(self.__mines[i][j] == 1):
                    self.L_mines[mine_cnt].setGeometry(QtCore.QRect(i * self.Icon_x, j * self.Icon_y, self.Icon_x, self.Icon_y))
                    mine_cnt += 1

        # Paint Minefield-Target


        self.L_target.setGeometry(QtCore.QRect(self.__target[0] * self.Icon_x, self.__target[1] * self.Icon_y, self.Icon_x, self.Icon_y))


        # Paint Minefield-Tank

        for i in range(self.__agent_num):
            self.L_tank[i].setGeometry(QtCore.QRect(self.__current[i][0] * self.Icon_x, self.__current[i][1] * self.Icon_y, self.Icon_x,self.Icon_y))
            self.L_tank[i].setPixmap(self.Tankicon[self.__currentBearing[i]])

        # 设置信息面板为空
        #刷新面板
        QtWidgets.QApplication.processEvents()

        if(self.graphic):
            self.update()

    def main(self):
        str = input('请输入: ')
        if(str == '1'):
            self.doReset(self.m,self.agent)

if __name__ == '__main__': #如果整个程序是主程序
     # QApplication相当于main函数， 也就是整个程序（很多文件）的主入口函数。
     # 对于GUI程序必须至少有一个这样的实例来让程序运行。

    app = 0
    app = QtWidgets.QApplication(sys.argv)
    #生成 mywindow 类的实例。
    window = mywindow(1, Maze(1))
    #有了实例，就得让它显示，show()是QWidget的方法，用于显示窗口。
    window.show()
    # 调用sys库的exit退出方法，条件是app.exec_()，也就是整个窗口关闭。
    # 有时候退出程序后，sys.exit(app.exec_())会报错，改用app.exec_()就没事
    # https://stackoverflow.com/questions/25719524/difference-between-sys-exitapp-exec-and-app-exec
    sys.exit(app.exec_())