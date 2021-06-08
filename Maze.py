# import java.awt.*
# import java.awt.event.*
# import javax.swing.*
# import java.io.*
import random
#-*- coding:utf-8 -*-
class Maze():

    def __init__(self, ag_num ):
        '''构造函数'''
            #self.size=200
        self.size = 16  #地图大小,Final修饰的成员变量终身不变，并且必须提前赋值
        self.numMines = 10 #雷的数量
        self.binarySonar = False #？？声纳是否是二进制？？

        self.LINEAR = 0 #线性
        self.EXP = 1 #指数
        self.RewardType = self.EXP  #self.LINEAR,"__变量名"表示为私有变量
        # RwardYype是线性(0)的还是指数型的(1)

        self.__agent_num = ag_num #Agent的数量
        self.__current = []
        self.__prev_current = []
        self.__target = []
        self.__currentBearing = [] #现在的方位
        self.__prev_bearing = []#之前的方位
        self.__targetBearing = []
        self.__sonar = [] #声纳的坐标吧(x,y)
        self.__av_sonar = []
        self.__range = []
        self.__mines = []

        self.__avs = []
        self.__end_state = []
        self.__conflict_state = []
        self.refreshMaze( ag_num )

    def get_agentnum(self):
        return self.__agent_num
    def set_conflict(self, i, j):#对 agent i 用agent j 设为发生冲突而停止的 Agent
        self.__avs[self.__current[i][0]][self.__current[i][1]] = 0 #这应该是把这个发生冲突的坐标置为0,表示在这个坐标上没有Agent了（如果有Agent 则avs[x][y]=Agent的编号）
        self.__end_state[i] = True #把 i 和 j 设置为已经停止
        self.__end_state[j] = True
        self.__conflict_state[i] = True #这一行暂时不动，为啥要设置两个 一个 self.__end_state 一个 conflict_state？
        self.__conflict_state[j] = True
 #        self.__current[i][0] = -1
 #        self.__current[i][1] = -1
 #        self.__current[j][0] = -1
 #        self.__current[j][1] = -1


    def check_i_conflict(self, i):#检查是否有与Agent i 冲突的Agent
        k = 0

        if ( self.__current[i][0] == self.__target[0] ) and ( self.__current[i][1] == self.__target[1] ) : #如果Agent i的当前状态已经到达目的地，那么就不会冲突了
            return  False
        if  self.__conflict_state[i]: #如果Agent i 已经被标记为停止了，就直接返回True
            return True
        if ( self.__current[i][0] < 0 ) or ( self.__current[i][1] < 0 ): #如果Agent i 当前的坐标为负数，那么代表就不会冲突
            return False
        for k in range(self.__agent_num): #遍历所有的 Agent
            if k == i : #自己不会与自己冲突
                continue
            if (self.__current[k][0] == self.__current[i][0] ) and ( self.__current[k][1] == self.__current[i][1]) : #如果两个 Agent 的坐标相等
                self.set_conflict( i, k ) #那么就把 i 和 j 两个Agent设置为冲突
                return True
        return False

    def check_conflict(self, agt, pos, actual ):#这里又重载了一个函数，在pos[0],pos[1]是否有冲突，这个坐标处这里的actual是判断是不是真正移动，是virtual_move or actual move
        k = 0
         #我猜测应该是 Agt这个agent在pos[0]pos[1]这个位置，检查有么有其他的Agent也在这个位置
        for k in range(self.__agent_num):
            if k == agt :
                continue
            if ( self.__current[k][0] == pos[0] ) and ( self.__current[k][1] == pos[1] ) :
                if actual : #如果是真的移动那就意味着两个agent发生撞击了
                    self.set_conflict( agt, k )
                    return True
        return False

    def refreshMaze(self, agt):
        '''#更新迷宫'''

        k = w = 0
        x = y = 0

         # limit the agent number between 1 and 10
        if (agt < 1):
             self.__agent_num = 1
        elif( agt > 100 ):
             self.__agent_num = 100
        else :
             self.__agent_num = agt

        self.__current = [([0] * 2) for i in range(self.__agent_num)]
        self.__target = [0,0]
        self.__prev_current = [([0] * 2) for i in range(self.__agent_num)]
        self.__currentBearing = [0] * self.__agent_num #所有Agent的当前方向
        self.__prev_bearing = [0] * self.__agent_num #所有Agnet之前的方向
        self.__targetBearing = [0] * self.__agent_num #所有Agent期望的目标方向
        self.__avs = [([0] * self.size) for i in range(self.size)]
        self.__mines = [([0] * self.size) for i in range(self.size)]
        self.__end_state =  [False] * self.__agent_num #判断Agent是否已经停止了
        self.__conflict_state = [False] *  self.__agent_num #判断Agent是否发生了冲突

        self.__sonar =[([0] * 5) for i in range(self.__agent_num)] #先初始化第一维
        self.__av_sonar = [([0] * 5) for i in range(self.__agent_num)]


        for k in range(3):
            d = random.random()   #返回一个随机数，[0.0,1.0]

        for k in range(self.__agent_num): #给每个Agent都随机生成一个初始位置
            while True:
                x = random.randint(0, self.size - 1)
                self.__current[k][0] = x
                y = random.randint(0, self.size - 1)
                self.__current[k][1] = y
                if(self.__avs[x][y] == 0):
                    self.__avs[x][y] = k + 1 #在地图上标出来Agent的位置
                    break

            for w in range(2):
                self.__prev_current[k][w] = self.__current[k][w] #之前的位置状态也标记成这个，反正是初始化无所谓的

            self.__end_state[k] = False
            self.__conflict_state[k] = False

        while True:
            x = random.randint(0, self.size - 1)
            self.__target[0] = x
            y = random.randint(0, self.size - 1)
            self.__target[1] = y
            if(self.__avs[x][y] == 0):
                break

        for _ in range(self.numMines):  #初始化雷的位置
            while True:
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)

                if ( self.__avs[x][y] == 0 ) and ( self.__mines[x][y] == 0 ) and ( x != self.__target[0] or y != self.__target[1] ) :
                    self.__mines[x][y] = 1
                    break
        for a in range(self.__agent_num): #禁止套娃！！调整所有Agent的方向为朝向旗子的一侧，只初始为上下左右
            self.setCurrentBearing( a, self.adjustBearing( self.getTargetBearing( a ) ) )
            self.__prev_bearing[a] = self.__currentBearing[a]
        print(self.__mines)
    def adjustBearing(self, old_bearing ):
        if( ( old_bearing == 1 ) or ( old_bearing == 7 ) ):
            return 0  #右上左上都归为上
        if( ( old_bearing == 3 ) or ( old_bearing == 5 ) ):
            return 4  #右下左下都归为下
        return old_bearing

    def getTargetBearing(self, i):    #获得目标方位

        if ( self.__current[i][0] < 0 ) or ( self.__current[i][1] < 0 ) :
            return 0
        #  [] d = new [ self.__agent_num]
        d = [0] * 2
        d[0] = self.__target[0] - self.__current[i][0]
        d[1] = self.__target[1] - self.__current[i][1]

        if( d[0] == 0 and d[1] < 0 ): #我是以左上角为坐标系向下向右建立坐标轴
            return( 0 ) #向上
        if( d[0] > 0 and d[1] < 0 ):
            return( 1 ) #右上
        if( d[0] > 0 and d[1] == 0 ):
            return( 2 ) #右
        if( d[0] > 0 and d[1] > 0 ):
            return( 3 ) #右下
        if( d[0] == 0 and d[1] > 0 ):
            return( 4 ) #下
        if( d[0] < 0 and d[1] > 0 ):
            return( 5 ) #左下
        if( d[0] < 0 and d[1] == 0 ):
            return( 6 ) #左
        if( d[0] < 0 and d[1] < 0 ):
            return( 7 ) #左上
        return( 0 )


    def get_all_TargetBearing(self):#又重载了一个获取目标方位的函数，这个是直接返回所有Agent的目标方位

        ret = [0] * self.__agent_num
        k = 0

        for k in range(self.__agent_num):
            ret[k] = self.getTargetBearing( k )
        return ret

    def getCurrentBearing(self, i ): #获取 Agent i的当前方位
        return  self.__currentBearing[i]

    def get_all_CurrentBearing(self):  #获取所有Agent的当前方位
        return ( self.__currentBearing )

    def setCurrentBearing(self, i, b ):#把 Agent i 的方位设成 b
        self.__currentBearing[i] = b

    def set_all_CurrentBearing(self, b):
        self.__currentBearing = b #这种直接赋值的方法非常不安全，因为b如果被删除了CurrentBearing也就没了
        #  推荐使用 Syetem.arraycopy(b,0,self.__currentBearing,0,b.length)

    def getReward(self, agt, pos, actual, immediate):#immediate应该是是否是即时奖励
     #这个函数应该是获得agt当前的奖励值
        x = pos[0]
        y = pos[1]

        if ( x == self.__target[0] ) and ( y == self.__target[1] ) : # reach self.__target
            self.__end_state[agt] = True
            self.__avs[x][y] = 0 #agt从地图上消失了，他所在的位置可以通过了
            return 1.0 #获得奖励1
        if ( x < 0 ) or ( y < 0 )  :# out of field
            return -1.0
        if self.__mines[x][y] == 1 :      # hit self.__mines
            return  0.0
# >>>>>>>>>#这里去除了注释，我感觉如果检测到agt，在这个pos[0]pos[1]位置上，已经发生冲突停止了，那么应该不会获得奖励
        if self.check_conflict(agt, pos, actual) :
           return  0.0
# >>>>>>>>>>>如果程序出错了，那么可能就是因为这里
        if(immediate): #如果不是即时奖励，那么一个trial结束后就不会得到基于与target的距离的Reward
            if (self.RewardType==self.LINEAR):
                r = self.getRange( agt )
                if (r > 10):
                    r = 10
                return( 1.0 - r /10.0 )  #adjust ermediate reward Reward要限定在[0,1]之间
            else : #utility=1/(1+rd)
                return(1.0/( 1 + self.getRange( agt ) ))   #adjust ermediate reward
        return (0.0)    #no ermediate reward


    def get_i_Reward(self,i, immediate ):#获取agent i 的奖励
        return( self.getReward( i, self.__current[i], True, immediate ) )


    def get_all_Reward(self, immediate) :
        k = 0
        r = [0.0] * self.__agent_num
        for k in range(self.__agent_num):
            r[k] = self.get_i_Reward( k, immediate )
        return( r )

    def get_a_b_Range(self, a, b ): #两个点 a和 b 返回两者x和y坐标相差较大的那一段距离
        range = 0
        d = [0] * 2

        d[0] = abs( a[0] - b[0] )
        d[1] = abs( a[1] - b[1] )
        range = max( d[0], d[1] )
        return( range )

    def getRange(self,i):  #返回 I 和 self.__target 坐标相差较大的那一段距离
       return( self.get_a_b_Range( self.__current[i], self.__target ) )


    def get_i_j_Range(self,i,j):#返回 I 和 j 相差较大的那个坐标轴的距离
        return(self.get_a_b_Range( self.__current[i], self.__current[j] ) )

    def get_all_Range(self):#返回所有agent与target的相差较大的那个坐标轴的距离
        k = 0
        all_range = [0] * self.__agent_num
        for k in range(self.__agent_num):
            all_range[k] = self.getRange( k )
        return( all_range )


    def getTargetRange(self, i):
        # return (1.0 / (1 + self.getRange( i )))
        return (1.0 / (1 + self.getRange( i )))

    def get_all_TargetRange(self):
        k = 0
        range = [0] * self.__agent_num
        for k in range(self.__agent_num):
            range[k] = self.getTargetRange( k )
        return(range)

    def getSonar(self, agt, new_sonar ):
        r = 0
        x = self.__current[agt][0]
        y = self.__current[agt][1]
        if ( x < 0 ) or ( y < 0 ) :
            for k in range(5):
                new_sonar[k] = 0
            return

        aSonar =  [0.0] * 8 #八个方位输入

        r = 0 #r就是当前位置(x,y)距离墙或者雷的距离
        while (y - r >= 0) and (self.__mines[x][y-r] != 1):    #从(x,y)位置向上摸索，看看有没有雷或者墙
            r = r + 1
        if r == 0 : # or y-r<0) #也就是说在（x,y）上有颗雷，这时候显然就不能有输入了
            aSonar[0] = 0.0
        else : #
            aSonar[0] = 1.0 / r

        r = 0
        while (x + r <= self.size - 1) and (y - r >= 0) and (self.__mines[x+r][y-r] != 1) :
            r = r + 1
        if r == 0:
            aSonar[1] = 0.0
        else :
            aSonar[1] = 1.0 / r

        r = 0
        while (x + r <= self.size - 1 and self.__mines[x+r][y] != 1):
            r = r + 1
        if (r == 0):
            aSonar[2] = 0.0
        else :
            aSonar[2] = 1.0 / r

        r = 0
        while (x + r <= self.size - 1 and y + r <= self.size - 1 and self.__mines[x+r][y+r] != 1):
            r = r + 1
        if (r == 0):
            aSonar[3] = 0.0
        else :
            aSonar[3] = 1.0 / r

        r = 0
        while (y + r <= self.size - 1 and self.__mines[x][y+r] != 1):
            r = r + 1
        if (r==0) :
            aSonar[4] = 0.0
        else :
            aSonar[4] = (1.0 / r)

        r=0
        while (x-r>=0 and y+r<=self.size-1 and self.__mines[x-r][y+r]!=1):
            r = r + 1
        if (r==0) :
            aSonar[5] = 0.0
        else :
            aSonar[5] = 1.0 / r

        r=0
        while (x-r>=0 and self.__mines[x-r][y]!=1):
            r = r + 1
        if (r==0) :
            aSonar[6] = 0.0
        else :
            aSonar[6] = 1.0 / r

        r=0
        while (x-r>=0 and y-r>=0 and self.__mines[x-r][y-r]!=1):
            r = r + 1
        if (r==0) :
            aSonar[7] = 0.0
        else :
            aSonar[7] = 1.0 / r

        self.__currentBearing = self.get_all_CurrentBearing ()

        for k in range(5):
            new_sonar[k] = aSonar[(self.__currentBearing[agt] + 6 + k) % 8] #这也太绕了我靠，new_sonar的方位是顺时针，从左方向开始计数，左方向为0 右方向为4，aSonar的方位是从上方向开始的 左侧为6 右侧为2
            if (self.binarySonar):#上面那式子就是做一个转换，把all_Sonar的八个方向的五个方向取过来放到new_sonar中
                if (new_sonar[k] < 1):
                    new_sonar[k] = 0  # binary self.__sonar signal
        return


    def getAVSonar(self, agt, new_av_sonar ): #获取视野内的agent的距离

        r = 0
        x = self.__current[agt][0]
        y = self.__current[agt][1]

        if( ( x < 0 ) or ( y < 0 ) ):

            for k in range(5):
                new_av_sonar[k] = 0 #初始化当前Agent的五个感知信号的输入
            return


        aSonar = [0] * 8 #初始化八个方向的探测Agent的声纳信号

        r=0
        while( y-r>=0 and (self.__avs[x][y-r]==(agt+1) or self.__avs[x][y-r]==0) ): #y-r>=0限制有没有到墙边，程序里的Agent的编号是0-7，实际编号为1-8，这里的Agt+1就是指本身，向上探测，看是否有其余的Agent
            r = r + 1
        if (r==0) :
            aSonar[0] = 0.0 #
        else :
            aSonar[0] = 1.0 / r

        r=0
        while (x+r<=self.size-1 and y-r>=0 and ( self.__avs[x+r][y-r]==(agt+1) or self.__avs[x+r][y-r]==0 ) ): #右上
            r = r + 1
        if (r==0) :
            aSonar[1] = 0.0
        else :
            aSonar[1] = 1.0 / r

        r=0
        while (x+r<=self.size-1 and ( self.__avs[x+r][y]==(agt+1) or self.__avs[x+r][y]==0 ) ):#右侧
            r = r + 1
        if (r==0) :
            aSonar[2] = 0.0
        else :
            aSonar[2] = 1.0 / r

        r=0
        while (x+r<=self.size-1 and y+r<=self.size-1 and ( self.__avs[x+r][y+r]==(agt+1) or self.__avs[x+r][y+r]==0 ) ): #右下
            r = r + 1
        if (r==0) :
            aSonar[3] = 0.0
        else :
            aSonar[3] = 1.0 / r

        r=0
        while (y+r<=self.size-1 and ( self.__avs[x][y+r]==(agt+1) or self.__avs[x][y+r]==0 ) ):#下
            r = r + 1
        if (r==0) :
            aSonar[4] = 0.0
        else :
            aSonar[4] = 1.0 / r

        r=0
        while (x-r>=0 and y+r<=self.size-1 and ( self.__avs[x-r][y+r]==(agt+1) or self.__avs[x-r][y+r]==0 ) ):#左下
            r = r + 1
        if (r==0) :
            aSonar[5] = 0.0
        else :
            aSonar[5] = 1.0 / r

        r=0
        while (x-r>=0 and ( self.__avs[x-r][y]==(agt+1) or self.__avs[x-r][y]==0 ) ):#左
            r = r + 1
        if (r==0) :
            aSonar[6] = 0.0
        else :
            aSonar[6] = 1.0 / r

        r=0
        while (x-r>=0 and y-r>=0 and ( self.__avs[x-r][y-r]==(agt+1) or self.__avs[x-r][y-r]==0 ) ):#左上
            r = r + 1
        if (r==0) :
            aSonar[7] = 0.0
        else :
            aSonar[7] = 1.0 / r

        self.__currentBearing = self.get_all_CurrentBearing ()

        for k in range(5):
            new_av_sonar[k] = aSonar[(self.__currentBearing[agt]+6+k)%8] #方向转换，这里的+6要谨慎对待，可以替换为-2，不过不同的编译器对于负数求余的认知不同，还是尽量使用正数求余
            if( self.binarySonar ): #二值化输入的声纳信号，只有0和1
                if( new_av_sonar[k] < 1 ):#
                    new_av_sonar[k] = 0  # binary self.__sonar signal

        return

     #这个virtual_move函数用于虚拟执行下一步的行走以计算奖励，不实际改变方向和坐标，而把虚拟行走后的坐标存入res[0]andres[1]中
    def virtual_move(self, agt, d, res ):#Agent的虚拟行走函数，一次行走1，agt为Agent的编号，d为要前进的方向（d为相对方向，d的取值应为-2 -1 0 1 2），res为Agt对于Agent的坐标

        bearing = ( self.__currentBearing[agt] + d + 8 ) % 8 #计算按d行走后的绝对方向

        res[0] = self.__current[agt][0]
        res[1] = self.__current[agt][1]


        if bearing == 0:
            if( res[1] > 0 ):
                res[1] -= 1

        elif bearing == 1:
            if( ( res[0] < self.size - 1 ) and ( res[1] > 0 ) ):
                res[0] += 1
                res[1] -= 1

        elif bearing == 2:
            if( res[0] < self.size - 1 ):
                res[0] += 1

        elif bearing == 3:
            if( ( res[0] < self.size - 1 ) and ( res[1] < self.size - 1 ) ):

                res[0] += 1
                res[1] += 1
        elif bearing == 4:
            if( res[1] < self.size - 1 ):
                res[1] += 1

        elif bearing == 5:
            if( ( res[0] > 0 ) and ( res[1] < self.size - 1 ) ):
                res[0] -= 1
                res[1] += 1

        elif bearing == 6:
            if( res[0] > 0 ):
                res[0] -= 1

        elif bearing == 7:
            if( ( res[0] > 0 ) and ( res[1] > 0 ) ):
                res[0] -= 1
                res[1] -= 1
        else:
            pass

        return
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def turn(self, i, d ):#转向
        bearing = self.getCurrentBearing( i )
        bearing = ( bearing + d ) % 8
        self.setCurrentBearing( i, bearing )


    def move(self, i, d ):#这是Agent实际的移动函数,d为相对方向，移动成功就返回1，移动不成功就返回-1 
        k = 0

        if( ( self.__current[i][0] < 0 ) or ( self.__current[i][1] < 0 ) ):
            return( -1 )

        for k in range(2):
            self.__prev_current[i][k] = self.__current[i][k]

        self.__prev_bearing[i] = self.__currentBearing[i]

        self.__currentBearing[i] = ( self.__currentBearing[i] + d + 8 ) % 8


        if(self.__currentBearing[i]== 0):
            if (self.__current[i][1] > 0): self.__current[i][1] -= 1
            else:
#               turn( i )
                return( -1 )

        elif(self.__currentBearing[i]== 1):
            if (self.__current[i][0] < self.size - 1 and self.__current[i][1] > 0):
                self.__current[i][0] += 1
                self.__current[i][1] -= 1

            else :
#               turn( i )
                return( -1 )

        elif(self.__currentBearing[i]==  2):
            if (self.__current[i][0]<self.size-1): self.__current[i][0] += 1
            else :
#                turn( i )
                return( -1 )


        elif(self.__currentBearing[i]==  3):
            if (self.__current[i][0]<self.size-1 and self.__current[i][1]<self.size-1):
                self.__current[i][0] += 1
                self.__current[i][1] += 1

            else :
#               turn( i )
                return( -1 )


        elif(self.__currentBearing[i]==  4):
            if (self.__current[i][1]<self.size-1):
                self.__current[i][1] += 1
            else :
#               turn( i )
                return( -1 )


        elif(self.__currentBearing[i]==  5):
            if (self.__current[i][0]>0 and self.__current[i][1]<self.size-1):
                self.__current[i][0] -= 1
                self.__current[i][1] += 1

            else :
#               turn( i )
                return( -1 )


        elif(self.__currentBearing[i]==  6):
            if (self.__current[i][0]>0): self.__current[i][0] -= 1
            else :
#               turn( i )
                return( -1 )


        elif(self.__currentBearing[i]==  7):
            if (self.__current[i][0]>0 and self.__current[i][1]>0):
                self.__current[i][0] -= 1
                self.__current[i][1] -= 1

            else :
#               turn( i )
                return( -1 )


        else:
            pass

        self.__avs[self.__prev_current[i][0]][self.__prev_current[i][1]] = 0
        self.__avs[self.__current[i][0]][self.__current[i][1]] = i + 1

        return (1)


    # return True if the move still keeps the agent within the field 检查是否超出边界
    def withinField (self, i, d ):  #检测AGent i 还能否在 相对方向d 上继续移动

        testBearing = ( self.__currentBearing[i] + d + 8 ) % 8
        if testBearing == 0:
            if (self.__current[i][1]>0):
                return (True)

        elif testBearing == 1:
            if (self.__current[i][0]<self.size-1 and self.__current[i][1]>0):
                return( True )

        elif testBearing == 2:
            if (self.__current[i][0]<self.size-1): return (True)

        elif testBearing == 3:
            if (self.__current[i][0]<self.size-1 and self.__current[i][1]<self.size-1):
                return( True )

        elif testBearing == 4:
            if (self.__current[i][1]<self.size-1):
                return( True )

        elif testBearing == 5:
            if (self.__current[i][0]>0 and self.__current[i][1]<self.size-1):
                return (True)

        elif testBearing == 6:
            if (self.__current[i][0]>0):
                return( True )

        elif testBearing == 7:
            if (self.__current[i][0]>0 and self.__current[i][1]>0):
                return( True )

        else:
            pass

#       System.out.prln ( "OutOfField: self.__current = ("+self.__current[i][0]+","+self.__current[i][1]+")  testBearing = " + testBearing)
        return (False)


    def move_all( self, d ): #一次移动所有的Agent

        k = 0

        res = [0] * self.__agent_num
        for k in range(self.__agent_num):
            res[k] = self.move( k, d[k] )
        return res


    def undoMove(self): #取消上次的移动
        self.__currentBearing = self.__prev_bearing
        self.__current[0] = self.__prev_current[0]
        self.__current[1] = self.__prev_current[1]


    def nextReward(self, agt, d, immediate ):

        r = 0.0
        next_pos = [0] * 2

        self.virtual_move( agt, d, next_pos ) #next_pos接收虚拟移动后agt的坐标
        r = self.getReward( agt, next_pos, False, immediate )  #consider revise
         # self.undoMove()
        return r


    def endState(self, agt ):#修改并返回agt的end_state 用于判断当前agt是否已经停止

        x = self.__current[agt][0]
        y = self.__current[agt][1]

        if( self.__conflict_state[agt] ):#agt已经发生冲突了

            self.__end_state[agt] = True
            return( self.__end_state[agt] )

        if( ( x < 0 ) or ( y < 0 ) ):#出界了

            self.__end_state[agt] = True
            return( self.__end_state[agt] )

        if( ( x == self.__target[0] ) and ( y == self.__target[1] ) ):#到达终点了

            self.__end_state[agt] = True
            self.__avs[x][y] = 0
            return( self.__end_state[agt] )

        if( ( self.__mines[x][y] == 1 ) or ( self.check_i_conflict( agt ) ) or ( self.__end_state[agt] ) ):# 踩雷or检测冲突oragt已经停止

            self.__avs[x][y] = 0
            self.__end_state[agt] = True

        else :
            self.__end_state[agt] = False
        return( self.__end_state[agt] )


    def endState_target_moving(self,target_moving ):#这个函数用于检测 target_moving 模式下 是否所有的Agent都停止工作了，如果是的话返回True


        bl = True #这一参数是用来当target是移动的时候，返回Agent k 当前是否已经进入endState
        for k in range(self.__agent_num):

            if( target_moving ):
                if( self.isHitTarget( k ) ):
                    return( True )
                if( not self.endState( k ) ):#如果Agent k 还没有停止
                    bl = False

            else :
                if( not self.endState( k ) ):
                    return( False )

        if( target_moving ):
            return( bl )
        else :
            return( True )


    def endState_normal(self):#这个函数用于检测普通模式下是否所有的Agent都停止运动了，如果有Agent尚未！endState(k) 则 返回False

        k = 0

        for k in range(self.__agent_num):
            if( not self.endState( k ) ):
                return( False )
        return( True )


    def isHitMine(self, i ):#判断是否踩雷，踩雷就返回True

        if( ( self.__current[i][0] < 0 ) or ( self.__current[i][1] < 0 ) ):
            return( False )
        if( self.__mines[self.__current[i][0]][self.__current[i][1]] == 1 ):
            return True
        else :
            return False


    def isConflict(self, i): #判断i是否发生冲突

        return( self.__conflict_state[i] )


    def isHitTarget(self, i): #在target_move模式下判断是否已经到达目标

        if( ( self.__current[i][0] == self.__target[0] ) and ( self.__current[i][1] == self.__target[1] ) ):
            return True
        else :
            return False


    def test_mines(self, i, j ):#判断坐标(i,j)是否有雷

        if( self.__mines[i][j] == 1 ):
            return( True )
        else :
            return( False )


    def test_current( self, agt, i, j ):#判断agt当前的坐标是不是(i,j)
        if ( self.__current[agt][0] == i and self.__current[agt][1] == j ):
            return( True )
        else :
            return( False )


    def test_target(self,i,j ): #判断当前target的坐标是不是（i，j）

        if( ( self.__target[0] == i ) and ( self.__target[1] == j ) ):
            return( True )
        else :
            return( False )


    def getMines(self,i,  j ):#获得当前坐标是否有雷

        return( self.__mines[i][j] )


    def getCurrent(self, agt ): #获取当前agt的坐标

        return( self.__current[agt] )


    def get_all_Current(self):#获取当前所有agt的坐标

        return( self.__current )


    def getCurrent_to_path(self, agt, path ): #把agt的坐标存入path[]

        for k in range(2):
            path[k] = self.__current[agt][k]
        return


    def get_all_Current_to_path(self,path):#把所有agt的坐标存入path[][]


        for i in range(self.__agent_num):
            for j in range(2):
                path[i][j] = self.__current[i][j]
        return


    def getPrevCurrent(self, agt):

        return( self.__prev_current[agt] )


    def get_all_PrevCurrent(self):

        return( self.__prev_current )


    def getTarget(self):

        return( self.__target )


    def go_target(self):

        new_pos = [0] * 2
        b = 0

        # for _ in range(3):
        #     d = random.random() 
        while True:

            b = random.randint(0, self.size - 1) #在0-15随机去一个数，这里的范围大于8的原因是target不会每一个时刻都在动，可以
            self.virtual_move_target( b, new_pos )
            if( self.valid_target_pos( new_pos ) == True): #如果virtual_move_target移动后的结果满足要求，则就移动target
                break

        self.move_target( b )
        return


    def valid_target_pos(self, new_pos ):#判断这个坐标是否可用

        x = new_pos[0]
        y = new_pos[1]

        if( ( x < 0 ) or ( x >= self.size ) ):
            return( False )
        if( ( y < 0 ) or ( y >= self.size ) ):
            return( False )
        if( self.__avs[x][y] > 0 ):
            return( False )
        if( self.__mines[x][y] == 1):
            return( False )
        return( True )


    def virtual_move_target(self, d, new_pos ): #

        new_pos[0] = self.__target[0]
        new_pos[1] = self.__target[1]

        if d == 0:
            new_pos[1] -= 1

        elif d == 1:
            new_pos[0] += 1
            new_pos[1] -= 1

        elif d == 2:
            new_pos[0] += 1

        elif d == 3:
            new_pos[0] += 1
            new_pos[1] += 1

        elif d == 4:
            new_pos[1] += 1

        elif d == 5:
            new_pos[0] -= 1
            new_pos[1] += 1

        elif d == 6:
            new_pos[0] -= 1

        elif d == 7:
            new_pos[0] -= 1
            new_pos[1] -= 1

        else:
            pass



    def move_target(self,d ):

            if d== 0:
                self.__target[1] -= 1

            elif d == 1:
                self.__target[0] += 1
                self.__target[1] -= 1

            elif d == 2:
                self.__target[0] += 1

            elif d == 3:
                self.__target[0] += 1
                self.__target[1] += 1

            elif d == 4:
                self.__target[1] += 1

            elif d == 5:
                self.__target[0] -= 1
                self.__target[1] += 1

            elif d == 6:
                self.__target[0] -= 1

            elif d == 7:
                self.__target[0] -= 1
                self.__target[1] -= 1

            else:
                pass
    # def get_x_y(self):
    #     print(self.__current);
    #     print(self.__currentBearing);
    #     print(self.__avs);
    #     print(self.__av_sonar)
    #     print(self.__target)

if __name__ == "__main__":
    maze = Maze(10)
