from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *

from AGENT import AGENT
from Maze import Maze
from Falcon import FALCON
from BP1 import BP
from DQNN import DQN
from Mywindow import mywindow
from Ui_TD_Falcon import Ui_Form

import os
import sys
import time,datetime
import math
from math import sin,cos,pi,sqrt

class MNT:
    
    RFALCON =0  
    TDFALCON=1
    BPN     =2
    DQNMethod = 3
    QLEARNING=0 
    SARSA    =1
    TDMethod = QLEARNING
    #TDMethod=SARSA
    # AVTYPE = TDFALCON
    AVTYPE = DQNMethod
    # 
    # AVTYPE  = BPN
    #AVTYPE  =RFALCON

    __numRuns =1
    __interval =100

    graphic=True
    Track  =False
    Target_Moving=False
    Trace  = True

    __agent_num = 0
    agent = []

    def __init__(self):
        self.__step = 0
        self.__path = [[[]]]
        self.__maxTrial = 10
        self.__maxStep = 30
        self.__minStep = []
        self.__lastFlag = False
        self.sonar_num = 1 
        self.Delay = 0
        self.Bound = False
        self.ImmediateReward = True
        self.initMNT(1)

    def initMNT (self, agt):
        self.__agent_num = agt
        self.init_agent ()
        self.init_mnt ()
        self.init_parameters ()

    def init_agent (self):

        self.maze = Maze(self.__agent_num)
        self.agent = []

        if (self.AVTYPE==self.RFALCON):
            for k in range(self.__agent_num):
                self.agent.append(FALCON( k ))
            print("Agent Type: R-FALCON")

        elif (self.AVTYPE==self.TDFALCON):
            for k in range(self.__agent_num):
                self.agent.append(FALCON( k ))
            print("Agent Type: TD-FALCON")

        elif (self.AVTYPE==self.BPN):
            for k in range(self.__agent_num):
                self.agent.append(BP(k))
            print("Agent Type: BPN")

        elif (self.AVTYPE==self.DQNMethod):
            for k in range(self.__agent_num):
                self.agent.append(DQN(k))
            print("Agent Type: DQN")


    def init_parameters (self):
        for k in range(self.__agent_num):
            self.agent[k].setParameters (self.AVTYPE, self.ImmediateReward)

    def init_mnt(self):
        app = QtWidgets.QApplication(sys.argv)

        self.window = mywindow(self.__agent_num, self.maze)
        self.window.show()

        # Setting number of agents
        self.window.Agent_num.setText(str(self.__agent_num))

        # Setting number of trials
        self.maxTrial = 2000
        self.window.Trials.setText(str(self.maxTrial))

        # Setting maximum number of steps per trial
        self.maxStep = 30
        self.window.Steps.setText(str(self.maxStep))

        # Setting if the target is moving
        self.window.Target_Moving.stateChanged.connect(self.target_moving_checkbox)

        # Setting whether to display the path taken by agent
        self.window.Traced.stateChanged.connect(self.is_track)

        # Setting the time delay per step
        self.Delay = 0
        self.window.V1.setText(str(self.Delay))

        # Setting the reward scheme: immediate or delayed
        self.window.Immediate_Reward.stateChanged.connect(self.is_Immediate_Reward)

        # Setting the TD learning method
        self.window.Bounde_TD_Rule.stateChanged.connect(self.is_Bounde_TD_Rule)

        # Display Panels
        self.ImmediateReward = True
        self.Bound = True
        self.Track = False
        self.Target_Moving = False
        self.window.Immediate_Reward.setChecked(self.ImmediateReward)
        self.window.Bounde_TD_Rule.setChecked(self.Bound)
        self.window.Traced.setChecked(self.Track)
        self.window.Target_Moving.setChecked(self.Target_Moving)
        # Register listeners
        self.init_path()
        self.actionPerformed()
        sys.exit(app.exec_())

    def actionPerformed(self):
        self.window.Button_Reset.clicked.connect(self.doReset)
        self.window.Button_Step.clicked.connect(self.doStep)
        self.window.Button_Auto.clicked.connect(self.doAuto)
        # 未完待续

    def get_values(self):
        sTrial = self.window.Trials.displayText().strip()
        if( len(sTrial) > 0 ):
            self.__maxTrial =  int(sTrial)   
        else:
            self.__maxTrial = 0
        if( self.__maxTrial < 0 ):
            self.__maxTrial = 0

        if (self.__maxTrial > 50000):
            self.__interval = 5000
        elif (self.__maxTrial > 10000):
            self.__interval = 1000


        sStep = self.window.Steps.displayText().strip()
        if( len(sStep) > 0 ):
            self.__maxStep = int( sStep ) 
        else:
            self.__maxStep = 0
        if( self.__maxStep < 0 ):
            self.__maxStep = 0

        sDelay = self.window.V1.displayText().strip()
        if( len(sDelay) > 0 ):
            self.__Delay =  int( sDelay ) 
        else:
            self.__Delay = 0
        if( self.__Delay < 0 ):
            self.__Delay = 0
#                                   Part: Button-Reset
    def doReset(self):
        print("Reset Button")
        self.maze.refreshMaze(self.__agent_num)

        for agt in range(self.__agent_num):
            self.agent[agt].setPrevReward(0)

        self.init_path()
        self.window.doRefresh(self.maze)
        # 这里还缺少一个，把结果面板清空
        # p_msg.setMessage( "")
        r = [1.0] * self.__agent_num #暂时存储Agt的奖励
        r[agt] = self.maze.get_i_Reward( agt, self.ImmediateReward )
        if( r[agt] == 1.0 and self.Trace):
            print( "AV" + str(agt) + " Success: Target achieved!")

#                            Part: Step-Reset
    def doStep(self):
        self.get_values()   # 获取maxStep, Delay信息
        self.graphic = True
        self.window.graphic = True
        self.Trace = False
        for agt in range(self.__agent_num):
            self.agent[agt].setTrace( self.Trace )    # 设置智能体轨迹为true
        # for agt in range(self.__agent_num):
        #     if( ( not self.maze.endState_target_moving( self.Target_Moving ) ) and ( self.__step < self.__maxStep ) ): #没有达到终止状态且没有达到maxStep
        #         if( self.Target_Moving ):
        #             self.maze.go_target()
        #         self.do_i_Step( agt, self.maze.getRange( agt ), False ) #step by step
        #     else:
        #         if(self.maze.endState_target_moving( self.Target_Moving )):
        #             print("Agent" + str(agt) + "已经停止")
        #         elif(self.__step >= self.__maxStep ):
        #             print("Agent" + str(agt) + "已经到达最大步数")
        for agt in range(self.__agent_num):
            if( ( not self.maze.endState( agt ) ) and ( self.__step < self.__maxStep ) ): #没有达到终止状态且没有达到maxStep
                if( self.Target_Moving ):
                    self.maze.go_target()
                print("last",self.maze.getCurrentBearing( agt ))   
                self.do_i_Step( agt, self.maze.getRange( agt ), False )
                print("now",self.maze.getCurrentBearing( agt ))
                print(self.agent[agt].Input) #step by step
            else:
                if(self.maze.endState(agt)):
                    print("Agent" + str(agt) + "已经停止")
                elif(self.__step >= self.__maxStep ):
                    print("Agent" + str(agt) + "已经到达最大步数")
                #这里需要修改
        self.__step += 1
        self.window.doRefresh_Step(self.maze)

        r = [0] * self.__agent_num #暂时存储Agt的奖励
        r[agt] = self.maze.get_i_Reward( agt, self.ImmediateReward )
        if( r[agt] == 1.0 and self.Trace):
            print( "AV" + str(agt) + " Success: Target achieved!")
            # p_msg.setMessage( "AV" + agt + " Success: Target achieved!" )


    def do_i_Step(self, agt : int, lastReward : float, last : bool):
        PERFORM=0
        LEARN=1
        # INSERT=2

        type = 0     #0-fuzzART 1-ART2

        this_Q = 0.0
        max_Q=0.0
        new_Q=0.0
        this_Sonar = [0] * 5
        that_Sonar = [0] * 5
        this_AVSonar = [0] * 5
        that_AVSonar = [0] * 5

        # x, y, px, py = 0,0,0,0
        action = -1
        this_bearing = 0
        this_targetRange = 0.0

        if (self.Trace):
            print( "\nSelecting action ....")
        
        while(action == -1):
            self.maze.getSonar( agt, that_Sonar )   #获得mines和边界的声纳信息
            self.maze.getAVSonar( agt, that_AVSonar )   #autonomous vehicle (AV) 获得其他智能体和边界的声纳信息

            for k in range(5):
                this_Sonar[k] = that_Sonar[k]
                this_AVSonar[k] = that_AVSonar[k]

            this_bearing = ( 8 + self.maze.getTargetBearing( agt ) - self.maze.getCurrentBearing( agt ) ) % 8#获得方向
            # this_bearing = self.maze.getCurrentBearing( agt )
            this_targetRange = self.maze.getTargetRange( agt ) #获得目标范围

            self.agent[agt].setState( this_Sonar, this_AVSonar, this_bearing, this_targetRange )    #给self.agent置入状态
            if self.AVTYPE == self.DQNMethod:
                state = self.agent[agt].getState(this_Sonar, this_AVSonar, this_bearing, this_targetRange)
            if (self.agent[agt].direct_access): #方向是否可用,初始值为False
                action = self.agent[agt].doDirectAccessAction(agt, True, self.maze )   # action is from 0 to numAction
            else:
                if self.AVTYPE == self.DQNMethod:
                    action = self.agent[agt].doSelectValidAction(state, self.maze)
                else:

                    action = self.agent[agt].doSelectValidAction( True, self.maze )        # action is from 0 to numAction
            
            if (action == -1):
                self.agent[agt].turn(4)
                self.maze.turn(agt,4)
        # print(action)
        if (not self.maze.withinField (agt, action-2)):
            print( "*** Invalid action " + str(action) + " will cause out of field *** ")
        # print(action-2)
        v = self.maze.move( agt, action-2 ) # actual movement, self.maze direction is from -2 to 2
        r = 0.0
        
        if( v != -1 ):   # if valid move
            if (last==True and self.ImmediateReward==False):  #run out of time (without immediate reward)
                r = 0.0
            else:
                r = self.maze.get_i_Reward (agt,self.ImmediateReward) #获得奖励
            self.agent[agt].move (action-2, True)           # actually move, self.agent direction is from -2 to 2
            
        else:    # invalid move
            r = 0.0
            print( "*** Invalid action " + str(action) + " taken *** ")
            self.agent[agt].move (action-2, False)          # don't move, self.agent direction is from -2 to 2


        if (self.Trace and r==1.0):
            print( "Success")
        
        if (self.AVTYPE == self.DQNMethod):
            self.maze.getSonar( agt, that_Sonar )   #获得mines和边界的声纳信息
            self.maze.getAVSonar( agt, that_AVSonar )
            for k in range(5):
                this_Sonar[k] = that_Sonar[k]
                this_AVSonar[k] = that_AVSonar[k]
            this_bearing = ( 8 + self.maze.getTargetBearing( agt ) - self.maze.getCurrentBearing( agt ) ) % 8#获得方向
            # this_bearing = self.maze.getCurrentBearing( agt )
            this_targetRange = self.maze.getTargetRange( agt )
            next_state = self.agent[agt].getState(this_Sonar, this_AVSonar, this_bearing, this_targetRange)
            self.agent[agt].store_transition(state,action, r, next_state)
        else:
            # print("!!!!")
            #    Calculate new Q value from reward function if possible
            new_Q_value_assigned = True
            if (self.agent[agt].direct_access or self.ImmediateReward):
                if (r==1.0):
                    new_Q = 1.0
                elif(r==0.0):
                    new_Q = 0.0
                elif(self.ImmediateReward):
                    new_Q = r
                else:
                    new_Q_value_assigned = False

            else:
                new_Q_value_assigned = False
            #    Estimate new Q value through TD formula
            if (not new_Q_value_assigned):

                self.agent[agt].setAction( action)
                this_Q = self.agent[agt].doSearchQValue( PERFORM, type )

                new_sonar = [0] * 5
                self.maze.getSonar( agt, new_sonar )
                new_av_sonar = [0] * 5
                self.maze.getAVSonar( agt, new_av_sonar )

                new_target_bearing = self.maze.getTargetBearing( agt )
                new_current_bearing = self.maze.getCurrentBearing( agt )
                new_target_range = self.maze.getTargetRange( agt )
                
                # self.maze.eeee(agt)
                self.agent[agt].setState( new_sonar, new_av_sonar, ( 8 + new_target_bearing - new_current_bearing ) % 8, new_target_range )        
                max_Q = self.agent[agt].getMaxQValue(self.TDMethod, True, self.maze )

                # learn QValue for this state and action
                if(self.Bound==False):
                    new_Q = this_Q + self.agent[agt].QAlpha * ( r + self.agent[agt].QGamma * max_Q - this_Q )#Q-FALCON or S-FALCON
                    # thresholding - limit the Q value to 0 and 1
                    # new_Q = 1.0/(double) (1.0 + (double) Math.exp (-5*(new_Q-0.5)))

                    if (self.AVTYPE==self.TDFALCON):
                        if (new_Q<0):
                            new_Q = 0
                        if (new_Q>1):
                            new_Q = 1
                else:
                    new_Q = this_Q + self.agent[agt].QAlpha * ( r + self.agent[agt].QGamma * max_Q - this_Q ) * (1 - this_Q)#BQ-FALCON or BS-FALCON

                    if (new_Q<0 or new_Q>1):
                        print( "*** Bounded rule breached *** ") #违反Q值有界规则啊
                        print( "r = " + str(r) + " this_Q = " + str(this_Q) + " max_Q = " + str(max_Q) + " new_Q = " + str(new_Q))


            # Learning with state, action, and Q_value

            if (self.Trace):
                print ( "\nLearning state action value ....")

            self.agent[agt].setState( this_Sonar, this_AVSonar, this_bearing, this_targetRange ) #set back to old state
            self.agent[agt].setAction( action )
            self.agent[agt].setReward( new_Q )

            if (self.agent[agt].direct_access):
                self.agent[agt].doSearchAction( LEARN, type )
            else:
                self.agent[agt].doSearchQValue( LEARN, type )

            if (self.Trace):
                print( "Action = " + str(action) + " Reward = " + str(r) + " new_Q = " + str(new_Q) + " max_Q = " + str(max_Q))


            r = [0] * self.__agent_num #暂时存储Agt的奖励
            #更新每个Agt的方向和Target方向
            for agt in range(self.__agent_num):
                r[agt] = self.maze.get_i_Reward( agt, self.ImmediateReward )
                if( r[agt] == 1.0 and self.Trace):
                    print( "AV" + str(agt) + " Success: Target achieved!")
                    # p_msg.setMessage( "AV" + agt + " Success: Target achieved!" )

    def do_i_RStep(self, agt : int, lastReward : float, last : bool):
        # PERFORM=0
        LEARN=1
        # INSERT=2

        # mode = 0     #0-PERFORM 1-LEARN
        type = 0     #0-fuzzART 1-ART2

        r = 0.0
        action = 0
        this_Sonar = [0] * 5
        that_Sonar = [0] * 5
        this_AVSonar = [0] * 5
        that_AVSonar = [0] * 5

        this_bearing = 0
        this_targetRange = 0.0



        while(action == -1):
            self.maze.getSonar( agt, that_Sonar )   #获得mines和边界的声纳信息
            self.maze.getAVSonar( agt, that_AVSonar )   #autonomous vehicle (AV) 获得其他智能体和边界的声纳信息

            for k in range(5):
                this_Sonar[k] = that_Sonar[k]
                this_AVSonar[k] = that_AVSonar[k]

            this_bearing = ( 8 + self.maze.getTargetBearing( agt ) - self.maze.getCurrentBearing( agt ) ) % 8#获得方向
            this_targetRange = self.maze.getTargetRange( agt ) #获得目标范围

            self.agent[agt].setState( this_Sonar, this_AVSonar, this_bearing, this_targetRange )    #给self.agent置入状态
            self.agent[agt].initAction() #initialize action to all 1's
            self.agent[agt].setReward(1) #search for actions with good reward

            if (self.Trace):
                 print ("Sense and Search for an Action:")

            action = self.agent[agt].doDirectAccessAction(agt, True, self.maze )   # action is from 0 to numAction


            if (action == -1):    # No valid action deadend, backtrack(反向)，如果没有可选择的动作就把坦克反向
                print ( "*** No valid action, backtracking ***")
                self.agent[agt].turn(4)
                self.maze.turn(agt,4)

        if (self.Trace):
             print("Performing the Action:")
        if (not self.maze.withinField (agt, action-2)):
            print( "*** Invalid action " + str(action) + " will cause out of field *** ")

        v = self.maze.move( agt, action-2 ) # actual movement, self.maze direction is from -2 to 2

        if( v != -1 ):   # if valid move
            if (last==True and self.ImmediateReward==False):  #run out of time (without immediate reward)
                r = 0.0
            else:
                r = self.maze.get_i_Reward (agt,self.ImmediateReward) #获得奖励
            self.agent[agt].move (action-2, True)           # actually move, self.agent direction is from -2 to 2

        else:    # invalid move
            r = 0.0
            print( "*** Invalid action " + str(action) + " taken *** ")
            self.agent[agt].move (action-2, False)          # don't move, self.agent direction is from -2 to 2


        if (self.Trace and r==1.0):
            print( "Success")

        if(action != -1):
            self.maze.getSonar( agt, that_Sonar )   #获得mines和边界的声纳信息
            self.maze.getAVSonar( agt, that_AVSonar )   #autonomous vehicle (AV) 获得其他智能体和边界的声纳信息

            for k in range(5):
                this_Sonar[k] = that_Sonar[k]
                this_AVSonar[k] = that_AVSonar[k]

            this_bearing = ( 8 + self.maze.getTargetBearing( agt ) - self.maze.getCurrentBearing( agt ) ) % 8#获得方向
            this_targetRange = self.maze.getTargetRange( agt ) #获得目标范围

            self.agent[agt].setState( this_Sonar, this_AVSonar, this_bearing, this_targetRange )    #给self.agent置入状态
            self.agent[agt].setAction(action) #set action
            self.agent[agt].setReward(r)

            if(r > self.agent[agt].getPrevReward()):
                if(self.Trace):
                    print("\nLearn from positive outcome")
                self.agent[agt].setReward(1) #instead of r
                #actin = self.agent[agt].doSearchAction(LEARN, type) #learn current action lead to reward
                self.agent[agt].reinforce()
            else:
                if(r == 0 or r <= self.agent[agt].getPrevReward()):
                    if(self.Trace):
                        print("\nReset and Learn")
                    self.agent[agt].setReward(r)       # (or 1-r) marks as good action/
                    self.agent[agt].resetAction ()                         #seek alternative actions
                    action = self.agent[agt].doSearchAction(LEARN,type)    # learn alternative actions
                    self.agent[agt].penalize()

        r = [0] * self.__agent_num #暂时存储Agt的奖励
        #更新每个Agt的方向和Target方向
        for agt in range(self.__agent_num):
            r[agt] = self.maze.get_i_Reward( agt, self.ImmediateReward )
            if( r[agt] == 1.0 and self.Trace):
                print( "AV" + str(agt) + " Success: Target achieved!")
                # p_msg.setMessage( "AV" + agt + " Success: Target achieved!" )

    def init_path(self):
        self.__step = 0
        self.__path = [[[0,0] for i in range(self.__agent_num)] for i in range(self.__maxStep + 1)]
        self.__minStep = self.maze.get_all_Range()#更新每个Agt最短步数

        for agt in range(self.__agent_num):
        
            self.maze.getCurrent_to_path(agt, self.__path[self.__step][agt]) #把Agt的初始位置同步到path中
            
            self.agent[agt].init_path(self.__maxStep) #更新每个Agt内部的self.max_step step currentBearing self.__path
            #注意上一条将每个Agt的方向都初始化为0方向了

        self.window.setCurrentPath(self.maze, self.__path[self.__step], self.__step)
        self.__lastFlag = False #结束标志，即到了最大步数


    def doAuto(self): #自动

        sample = 0
        success = 0
        failure = 0
        time_out = 0
        conflict = 0

        agt = 0
        total_step = 0
        total_min_step = 0
        numCode = [0] * self.__agent_num
        reward = [0.0] * self.__agent_num
        result = False
        # NumberFormat nf = NumberFormat.getInstance() #返回当前默认语言环境的数字格式
        # nf.setMaximumFractionDigits (2) #小数显示最多位数超出四舍五入
        self.get_values() #从接口获取仿真参数

        if (self.__maxTrial > 1 or self.__numRuns>1):
            self.graphic=False
            self.window.graphic = False
            self.Trace = False
            for agt in range(self.__agent_num):
                self.agent[agt].setTrace( self.Trace )

        else:
            self.graphic=True
            self.window.graphic = True
            self.Trace = True
            for agt in range(self.__agent_num):
                self.agent[agt].setTrace(self.Trace )


        try:
              pw_score = open("score.txt", "w+",encoding = "utf-8")
              pw_avg = open("result.txt", "w+",encoding = "utf-8")
        except IOError:
            print("score.txt 或 result.txt 文件打开失败")

        numReadings = int(self.__maxTrial/self.__interval) + 1

        totalSuccess     = [0.0] * numReadings
        totalHitMine     = [0.0] * numReadings
        totalOutOfTime   = [0.0] * numReadings
        totalNSteps      = [0.0] * numReadings
        totalNCodes      = [0.0] * numReadings
        totalSqSuccess   = [0.0] * numReadings
        totalSqHitMine   = [0.0] * numReadings
        totalSqOutOfTime = [0.0] * numReadings
        totalSqNSteps    = [0.0] * numReadings
        totalSqNCodes    = [0.0] * numReadings

        totalSuccess[0]    =0.0
        totalHitMine[0]    =50.0
        totalOutOfTime[0]  =50.0
        totalNSteps[0]     =5.0
        totalNCodes[0]     =0.0
        totalSqSuccess[0]  =0.0
        totalSqHitMine[0]  =2500.0
        totalSqOutOfTime[0]=2500.0
        totalSqNSteps[0]   =25.0
        totalSqNCodes[0]   =0.0

        totalSteps  = 0
        # Date st = new Date ()
        # long start = st.getTime ()
        start = int(time.time()) #获取当前的时间戳

        for _ in range(self.__numRuns):
            rd=1
            trial = 0

            # if (self.__numRuns > 1):
            #     if (self.AVTYPE == self .RFALCON):
            #         for k in range(self.__agent_num):
            #             self.agent[k] = FALCON(k)

            #     elif (self.AVTYPE==self.TDFALCON):
            #         for k in range(self.__agent_num):
            #             self.agent[k] = FALCON(k)

            #     elif (self.AVTYPE==self.BPN):
            #         for k in range(self.__agent_num):
            #             self.agent[k] = BP(k)
                
            #     elif (self.AVTYPE==self.DQN):
            #         for k in range(self.__agent_num):
            #             self.agent[k] = BP(k)
                        

            #     for k in range(self.__agent_num):
            #         self.agent[k].setParameters (self.AVTYPE, self.ImmediateReward)
            print(self.__maxTrial)
            while( trial < self.__maxTrial ):

                self.maze.refreshMaze( self.__agent_num )
                self.window.doRefresh( self.maze )
                self.init_path()
                reward = self.maze.get_all_Reward(self.ImmediateReward)
                # print(trial)
                while( not self.maze.endState_target_moving( self.Target_Moving ) and self.__step < self.__maxStep ):

                    if( self.__step == ( self.__maxStep - 1 ) ):
                        self.__lastFlag=True

                    for agt in range(self.__agent_num):

                        self.agent[agt].setPrevReward (reward[agt])

                        if( self.maze.endState( agt )):
                            continue
                        if( self.Target_Moving ):
                            self.maze.go_target()
                        # print("step", self.__step)
                        if (self.AVTYPE == self.RFALCON):
                            self.do_i_RStep( agt, float( self.__minStep[agt] / ( self.__step + 1 ) ), self.__lastFlag )
                        else:
                            self.do_i_Step( agt, float( self.__minStep[agt] / ( self.__step + 1 ) ), self.__lastFlag )

                        if self.AVTYPE == self.DQNMethod:
                            self.agent[agt].EPISILO = self.agent[agt].EPISILO_END + (self.agent[agt].EPISILO_START - self.agent[agt].EPISILO_END) * \
                            math.exp(-1. * self.__step / self.agent[agt].EPISILO_DECAY)
                            if self.agent[agt].memory_counter >= self.agent[agt].BATCH_SIZE:
                                # print("Learn")
                                self.agent[agt].learn()

                        self.agent[agt].decay()
                    
                    self.__step += 1
                    self.maze.get_all_Current_to_path( self.__path[self.__step] )
                    self.window.setCurrentPath( self.maze, self.__path[self.__step], self.__step )
                    reward = self.maze.get_all_Reward(self.ImmediateReward)

                    self.window.setCurrent( self.maze )

                    self.window.setSonar(self.maze )
                    self.window.setBearing(self.maze )


                    if (self.graphic):
                        self.window.RePaint(self.maze)

                        time.sleep( self.__Delay)



                trial += 1
                totalSteps += self.__step

                if( not self.Target_Moving ):

                    for agt in range(self.__agent_num):

                        numCode[agt] = self.agent[agt].getNumCode()
                        if( reward[agt] == 1 ):

                            success += 1
                            total_step += self.__step
                            total_min_step += self.__minStep[agt]
                        elif( self.__step == self.__maxStep ):
                            time_out += 1
                        elif( self.maze.isConflict( agt ) ):
                            conflict += 1
                        else:
                            failure += 1


                    if( trial % self.__interval==0 ):
                        sample = self.__interval
                    else:
                        sample = trial % self.__interval
                    # p_msg.setMessage( "Success rate: " + success*100/(sample*self.__agent_num) + "%  Hit Mine: " + failure*100/(sample*self.__agent_num) + "%  Timeout: " + time_out*100/(sample*self.__agent_num) + "%  Collision: " + conflict*100/(sample*self.__agent_num) + "%" )
                    if(self.Trace):
                        print("Success rate: " + str(success * 100 / (sample * self.__agent_num)) + "%  Hit Mine: " + str(failure * 100 / (sample * self.__agent_num )) + "%  Timeout: " + str( time_out * 100 / (sample * self.__agent_num)) + "%  Collision: " + str(conflict * 100 / (sample * self.__agent_num)) + "%" )
                    if( trial % self.__interval==0 ):

                        success_rate = success * 100.0 / (sample*self.__agent_num)
                        failure_rate = failure * 100.0 / (sample*self.__agent_num)
                        time_out_rate = time_out*  100.0 / (sample*self.__agent_num)
                        if(total_min_step == 0): n_steps = 0
                        else: n_steps = total_step / float(total_min_step)
                        n_codes = numCode[0]

                        totalSuccess[rd]  += success_rate
                        totalHitMine[rd]  +=failure_rate
                        totalOutOfTime[rd]+=time_out_rate
                        totalNSteps[rd]   +=n_steps
                        totalNCodes[rd]   +=n_codes

                        totalSqSuccess[rd]  +=success_rate * success_rate
                        totalSqHitMine[rd]  +=failure_rate * failure_rate
                        totalSqOutOfTime[rd]+=time_out_rate * time_out_rate
                        totalSqNSteps[rd]     +=n_steps*n_steps
                        totalSqNCodes[rd]     +=n_codes*n_codes
                        rd  += 1

                        pw_score.write(str(trial) + " " + "%.2f"%(success*100/(sample*self.__agent_num)) + " " + "%.2f"%(failure*100/(sample*self.__agent_num)) + " " + "%.2f"%( time_out * 100 / (sample*self.__agent_num))  + " " + "%.2f"%(n_steps) + " " + str(numCode[0]) + "\n")

                        print( "Trial " + str(trial) + ": Success: " + "%.2f"%(success*100/(sample*self.__agent_num)) + "%  Hit Mine: " + "%.2f"%(failure*100/(sample*self.__agent_num)) + "%  Timeout: " + "%.2f"%(time_out*100/(sample*self.__agent_num)) + "% NSteps: " + " " +"%.2f"%(n_steps) + " NCodes: " + str(numCode[0]))

                        success = 0
                        failure = 0
                        time_out = 0
                        conflict = 0
                        total_step = 0
                        total_min_step = 0


                else:

                    result = False
                    for agt in range(self.__agent_num):

                        numCode[agt] = self.agent[agt].getNumCode()
                        if( reward[agt] == 1 ):

                            result = True
                            # p_msg.setMessage( "AV" + agt + " Success achieved in " + step + " steps" + " with " + numCode[agt] + " codes of agent " + agt );
                            print( "AV" + str(agt) + " Success achieved in " + str(self.__step) + " steps" + " with " + str(numCode[agt]) + " codes of self.agent " + str(agt) )
                            if (trial % self.__interval==0):
                                print( "AV" + str(agt) + " Success: Target achieved in " + str(self.__step) + " steps" + " with " + str(numCode[agt]) + " codes of self.agent " + str(agt) )


                    if( result ):
                        success += 1
                    else:

                        failure += 1
                        # p_msg.setMessage( "All fail!!!" )
                        print("All fail!!!")

                    if( trial % self.__interval==0 ):
                        sample = self.__interval
                    else:
                        sample = trial%self.__interval

                    # p_msg.setMessage( "Trial " + str(trial) + "  Success: " + "%.2f"%(success*100/sample) + "%  Failure: " + "%.2f"%(failure*100/sample) + "%" )
                    print( "Trial " + str(trial) + "  Success: " + "%.2f"%(success*100/sample) + "%  Failure: " + "%.2f"%(failure*100/sample) + "%" )

                    if( trial % self.__interval == 0 ):

                        pw_score.write( str(trial) + " " + "%.2f"%(success * 100 / sample) + " " + "%.2f"%(failure*100/sample)  + "\n")
                        success = 0
                        failure = 0



                # decay for Epsilon-greedy strategy

                for agt in range(self.__agent_num):
                    if (self.agent[agt].QEpsilon > self.agent[agt].minQEpsilon):
                        self.agent[agt].QEpsilon -= self.agent[agt].QEpsilonDecay


            if (self.AVTYPE==self.RFALCON or self.AVTYPE==self.TDFALCON):
                for agt in range(self.__agent_num):
                    self.agent[agt].saveAgent( "rule.txt")



        # Date et = new Date ()
        # long end = et.getTime ()
        end = int(time.time())

        avgTime  = (end-start)/(float)(self.__numRuns * self.__agent_num)  # per self.agent per experiment
        avgSteps = totalSteps/(float)(self.__numRuns * self.__agent_num)

        pw_avg.write("Trial" + "\t" + "SuccessRate" + "\t" + "StdDev" + "\t" +"HitMine" + "\t" + "Timeout" + "\t"+  "NormalizedStep" + "\t" + "NumberOfCodes"  + "\n")

        for i in range(numReadings):
            es  = totalSuccess[i] / self.__numRuns
            ess = totalSqSuccess[i] / self.__numRuns

            pw_avg.write (str(i*self.__interval) + "\t" + str(es) + "\t\t" + "%.2f"%(sqrt(ess - es*es)) + "\t\t" +"%.2f"%(totalHitMine[i]/self.__numRuns) + "\t\t" + "%.2f"%(totalOutOfTime[i]/self.__numRuns) + "\t\t"+  "%.2f"%(totalNSteps[i]/self.__numRuns) + "\t\t\t" + "%.2f"%(totalNCodes[i]/self.__numRuns) + "\n")


        pw_avg.write ("Average Time (msec)" + "\t : " + str(avgTime) + "\n")
        pw_avg.write("Average Number of Steps" + "\t : " + str(avgSteps) + "\n")
        pw_avg.write("Average Time per Step" + "\t : " + "%.2f"%(avgTime/avgSteps) + "\n")

        pw_score.close()
        pw_avg.close()

        self.Trace = True

# ——————————————————————————————分割线——————————————————————
    def target_moving_checkbox(self):
        if(self.window.Target_Moving.isChecked()):
            print("Target_Moving")
            self.Target_Moving = True
        else:
            self.Target_Moving = False
    def is_track(self):
        if(self.window.Traced.isChecked()):
            print("Tracked")
            self.Track =True
        else:
            self.Track = False
    def is_Immediate_Reward(self):
        if(self.window.Immediate_Reward.isChecked()):
            print("Immediate_Reward")
            self.ImmediateReward = True
        else:
            print("Delayed_Reward")
            self.ImmediateReward = False
    def is_Bounde_TD_Rule(self):
        if(self.window.Bounde_TD_Rule.isChecked()):
            print("Bounde_TD_Rule")
            self.Bound = True
        else:
            self.Bound = False


if __name__ == "__main__":
    k = MNT()
    # import res_rc
# 