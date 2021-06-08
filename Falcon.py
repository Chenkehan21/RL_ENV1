import random

from AGENT import AGENT
from Maze import Maze 

class FALCON(AGENT):

    __RFALCON = 0
    __TDFALCON = 1

    __forgetting = False
    __INTERFLAG  = False
    __detect_loop = False 
    __look_ahead = False
    __Trace = True 

    def __init__ (self, av_num) : #初始化函数
        super(FALCON,self).__init__()
        self.__numSpace=4  # 0-State 1-Action 2-Reward 3-New State 
        self.__numSonarInput=10
        self.__numAVSonarInput=10
        self.__numBearingInput=8
        self.__numRangeInput=0 
        self.__numAction=5
        self.__numReward=2
        self.__complementCoding=1
        self.__CURSTATE=0
        self.__ACTION  =1
        self.__REWARD  =2
        self.__NEWSTATE=3
        self.__FUZZYART=0  #模糊ART
        self.__ART2    =1
        self.__PERFORM =0
        self.__LEARN   =1
        self.__INSERT  =2

 
        self.__numCode = 0
        self.__prevReward = 0
        self.__activityF1 = [] #FAlcon的第一层，是一个二维，结点一共有三个输入Field，每个区域的大小分别为18，5，2
        self.__activityF2 = []#Falcon的第二层，是一个一维的(y1,y2,y3......)，大小为F2层所有的结点数
        self.__weight = []
        self.__J = 0#这个是啥玩意???
        self.__KMax = 3
        self.__newCode = []
        self.__confidence = []
        self.__initConfidence = 0.5
        self.__reinforce_rate = 0.5
        self.__penalize_rate = 0.2 #处罚率
        self.__decay_rate=0.0005 #衰减率
        self.__threshold=0.01 #阈值
        self.__capacity=9999 #最大的结点数
        self.__beta = 1.0 #β,这是ART权重参数W的学习率
        self.__epilson = 0.001 #ε，用于增大Field1 的警戒参数，使其略大于 Mj-ck1
        self.__gamma = [1.0, 1.0,1.0,0.0] #γ

        # Action enumeration，动作枚举

        self.__alpha = [0.1,0.1,0.1] #这个是αk，选择参数 choice parameters
        self.__b_rho = [0.2,0.2,0.5,0.0]  # fuzzy ART baseline vigilances  β_ρ  警戒参数
        self.__p_rho = [0.0,0.0,0.0,0.0]  # fuzzy ART performance vigilances  p_ρ ???这个参数目前还不知道什么意思

        self.__end_state = False
        self.__agentID = 0
        self.__max_step = 0
        self.__step = 0
        self.__currentBearing = 0
        self.__targetBearing = 0
        self.__path = []
        self.__current = []

        self.__agentID = av_num
        self.__numInput = [0] * self.__numSpace   # self.__numSpace:0-State 1-Action 2-Reward 3-New State 
        self.__numInput[0] = self.__numSonarInput+self.__numAVSonarInput+self.__numBearingInput+self.__numRangeInput #10+0+8+0
        self.__numInput[1] = self.__numAction 
        self.__numInput[2] = self.__numReward 
        self.__numInput[3] = self.__numInput[0]

        self.__activityF1 = []
        for i in range(self.__numSpace):
            self.__activityF1.append([0.0] * self.__numInput[i]) 

        self.__newCode = [True]

        self.__confidence = [0.0] * (self.__numCode+1)
        self.__confidence[0] = self.__initConfidence #self.__initConfidence=0.5,initConfidence是Q值的意思?

        self.__activityF2 = [0.0] * (self.__numCode+1) #第二层初始化一个unCommited的结点
        self.__weight = [[] for i in range(self.__numCode+1)]
        #权重向量[F2节点的index][self.__numSpace=4][]，W(ck)(j),ck中的k取值为1，2，3，指的是F1层的三个Field j为F2层的index 初始化只有一个uncommitted结点
        for j in range(self.__numCode + 1):
            for k in range(self.__numSpace):
                tmp = [1.0] * self.__numInput[k]
                self.__weight[j].append(tmp) #w[j][k][18,5,2,18]
            # for i in range(self.__numInput[k]):#对于uncommitted结点，每个[j][k][i]都为1
            #     self.__weight[j][k][i] = 1.0;

        self.__end_state = False #初始化endState=False

        self.__current = [0, 0] #初始化?? current是什么???


    def setParameters (self, AVTYPE, immediateReward):

        if (AVTYPE == self.__RFALCON): #如果用的是RFALCON，那么就不需要ε-greedy
            self.QEpsilonDecay = 0.00000
            self.QEpsilon      = 0.00000

        else :   #  QEpsilonDecay rate for TD-FALCON
            self.QEpsilonDecay = 0.00050
            self.QEpsilon      = 0.50000


        if (immediateReward): #如果是及时奖励,Discount factor γ=0.5
            self.QGamma =  0.5
        else :
            self.QGamma =  0.9


    def stop(self):#停止Falcon网络的更新
        self.__end_state = True


    def checkAgent (self, outfile): #outfile是文件名
        # PrintWriter pw_agent = null #PrintWriter Java用于写出的类
        # boolean invalid
        try:
            # pw_agent = new PrintWriter (new FileOutputStream(outfile),True)
            pw_agent = open(outfile, "a+")
        except IOError:  #用来捕获IO异常
            print("打开"+outfile+"文件失败")
        pw_agent.write("Number of Codes : " + str(self.__numCode) + "\n") #把结点的个数写入rule.txt

        for j in range(self.__numCode):
            invalid = False
            for i in range(self.__numInput[self.__ACTION]):
                if (self.__weight[j][0][i] == 1 and self.__weight[j][self.__ACTION][i] == 1):#这样的意思是该结点J代表在i方向上的State为1(离mines或墙很近)，并且还走这个方向的话，显然是很差劲的Node(因为一定会撞到雷上)
                    invalid = True

            if (invalid):
                pw_agent.write("Code " + str(j) + "\n")
                for k in range(self.__numSpace):
                    pw_agent.write("Space " + str(k) + " : ") #print不会换行,println会换行
                    for i in range(self.__numInput[k]):
                        pw_agent.write(str(self.__weight[j][k][i]) + ", ", end = '')
                    pw_agent.write("\n") #输出最后一位,并换行
        pw_agent.close ()


    def clean(self):#清除这些不好的Node
        numClean = 0
        for j in range(self.__numCode):
            for i in range(self.__numInput[self.__ACTION]):
                if (self.__weight[j][0][i] == 1 and self.__weight[j][self.__ACTION][i] == 1):
                    self.__newCode[j] = True #把j结点置为Uncommitted Node
                    numClean +=  1

        if (numClean > 0):
            print(str(numClean) + " bad code(s) removed.")


    def saveAgent (self, outfile):
        try:
            pw_agent = open(outfile, "w+")
        except IOError:
            print("打开"+outfile + "文件失败")

        pw_agent.write("Number of Codes : " + str(self.__numCode) + "\n")
        for j in range(self.__numCode+1 ):
            pw_agent.write ("Code " + str(j) + "\n")
            for k in range(self.__numSpace):
                pw_agent.write("Space " + str(k) + " : ")
                for i in range(self.__numInput[k]):
                    pw_agent.write(str(self.__weight[j][k][i]) + ", ")
                pw_agent.write("\n")
        pw_agent.close ()


    def getNumCode(self):
        return( self.__numCode )


    def getCapacity (self):
        return (self.__capacity)

    def setTrace (self, t):
        self.__Trace = t

    def setPrevReward (self, r):
        self.__prevReward = r

    def getPrevReward (self):
        return (self.__prevReward)

    def createNewCode (self):
        self.__numCode +=  1

        self.__activityF2 = [0.0] * (self.__numCode + 1)
        
        new_newCode = [False] * (self.__numCode + 1)
        for j in range(self.__numCode):
            new_newCode[j] = self.__newCode[j]
        new_newCode[self.__numCode] = True
        self.__newCode = new_newCode #这么更新的意义何在????? 这一步就是因为新建了一个uncommitted 结点要把新节点加入到newcode中来

        new_confidence = [0.0] * (self.__numCode + 1)
        for j in range(self.__numCode):
            new_confidence[j] = self.__confidence[j]
        new_confidence[self.__numCode] = self.__initConfidence
        self.__confidence = new_confidence #把新节点的Q值加入到confidence中

        new_weight = [[] for i in range(self.__numCode+1)]
        for j in range(self.__numCode):
            new_weight[j] = self.__weight[j]

        for k in range(self.__numSpace):
            tmp = [1.0] * self.__numInput[k]
            new_weight[self.__numCode].append(tmp)

        self.__weight = new_weight


    def reinforce (self):
        self.__confidence[self.__J]  +=  (1.0 - self.__confidence[self.__J])*self.__reinforce_rate #强化节点J，对于结点J，这个J被初始化为 int | Q = Q  +  (1 - Q) * α (α = 0.5)

    def penalize (self):
        self.__confidence[self.__J]  -=  self.__confidence[self.__J]*self.__penalize_rate #惩罚结点J，Q = Q - Q * self.__penalize_rate(self.__penalize_rate = 0.2)


    def decay (self):
        for j in range(self.__numCode):
            self.__confidence[j]  -=  self.__confidence[j]*self.__decay_rate #随着时间流逝，所有结点的可信度都会衰减


    def prune (self): #剪枝
        for j in range(self.__numCode):
            if (self.__confidence[j] < self.__threshold): #如果结点的可信度低于阈值，就把它取消，置为新节点
                self.__newCode[j] = True


    def purge (self):#清洗
        numPurge = 0 #用于计算F2层中newCode为True的结点个数
        for j in range(self.__numCode):
            if (self.__newCode[j] == True):
                numPurge +=  1
        if (numPurge > 0):
            new_weight = [[] for i in range(self.__numCode - numPurge + 1)] #这些下标后面都有一个 + 1就是为了添加那个一直存在的uncommitted Node
            new_newCode = [False] * (self.__numCode - numPurge + 1)
            new_confidence = [0.0]* (self.__numCode - numPurge + 1)

            print ("Total of " + str(self.__numCode) + " rule(s) created. ")
            k = 0
            for j in range(self.__numCode):
                if (self.__newCode[j] == False):
                    new_weight[k] = self.__weight[j]
                    new_newCode[k] = self.__newCode[j]
                    new_confidence[k] = self.__confidence[j]
                    k +=  1

            new_weight[self.__numCode - numPurge] = self.__weight[self.__numCode] #把最后一个Uncommitted Node 加上
            new_newCode[self.__numCode - numPurge] = self.__newCode[self.__numCode]
            new_confidence[self.__numCode - numPurge] = self.__confidence[self.__numCode]

            self.__weight = new_weight
            self.__newCode = new_newCode
            self.__confidence = self.__confidence
            self.__numCode  -=  numPurge #结点剪枝
            self.__activityF2 = [0.0]* (self.__numCode + 1)
            print(str(numPurge) + " rule(s) purged.")
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def setState(self,sonar = [], av_sonar = [], bearing = 0, target_range = 0.0 ):  #State的输入 10  +  0  +  (int)target的方向  +  0

        sonar_input = int(self.__numSonarInput / 2 )
        avsonar_input = int(self.__numAVSonarInput / 2)
        range_intput = int(self.__numRangeInput / 2)
        for i in range(sonar_input):  #输入10个Sonar的信号，numSonarInput = 10

            self.__activityF1[0][i] = sonar[i]
            self.__activityF1[0][i + sonar_input] = 1 - sonar[i]

        index = self.__numSonarInput

        for i in range(avsonar_input):

            self.__activityF1[0][index + i] = av_sonar[i]
            self.__activityF1[0][index + i + sonar_input] = 1 - av_sonar[i]

        index  +=  self.__numAVSonarInput #没有AvSonar的输入

        for i in range(self.__numBearingInput):
            self.__activityF1[0][index + i] = 0.0
        self.__activityF1[0][index + bearing] = 1.0
        index  +=  self.__numBearingInput

        for i in range( range_intput): #也没有RangeInput的输入

            self.__activityF1[0][index + i] = target_range
            self.__activityF1[0][index + i + range_intput] = 1 - target_range


    def initAction (self):  #ACtion向量初始化为1
        for i in range(self.__numInput[self.__ACTION]) :
            self.__activityF1[self.__ACTION][i] = 1

# ????
    def init_path(self, maxstep = 0): #path用来记录Agent行走的坐标
        '''在刷新地图后 更新当前Agt的路径记录，当前位置'''
        self.__max_step = maxstep
        self.__step = 0
        self.__path = [[0, 0] for i in range(self.__max_step + 1)]
        # self.__path[0] = current
        # self.__current = current

    def init_Bearing(self, currentBearing = 0,targetBearing = 0):
        self.__currentBearing = currentBearing
        self.__targetBearing = targetBearing

    def display_Brearing(self):
        print(self.__currentBearing)
        print(self.__targetBearing)

    def resetAction (self) : #这种Rest意义何在??,把动作的值反转
        for i in range(self.__numInput[self.__ACTION]) :
            self.__activityF1[self.__ACTION][i] = 1 - self.__activityF1[self.__ACTION][i]

    def setAction (self, action = 0):  #把当前动作设置为action
        for i in range(self.__numInput[self.__ACTION]):
            self.__activityF1[self.__ACTION][i] = 0
        self.__activityF1[self.__ACTION][action] = 1.0


    def setReward (self, r = 0.0):
        self.__activityF1[self.__REWARD][0] = r
        self.__activityF1[self.__REWARD][1] = 1 - r


    def initReward (self):  #Reward全部初始化为1
        self.__activityF1[self.__REWARD][0] = 1
        self.__activityF1[self.__REWARD][1] = 1


    def setNewState(self, sonar = [], av_sonar = [], bearing = 0, target_range = 0): #和SetState一样

        index = 0
        sonar_input = int(self.__numSonarInput / 2 )
        avsonar_input = int(self.__numAVSonarInput / 2)
        range_input = int(self.__numRangeInput / 2)

        for i in range(sonar_input):

            self.__activityF1[self.__NEWSTATE][i] = sonar[i]  #Newsate = 3 curstate = 0
            self.__activityF1[self.__NEWSTATE][i + sonar_input] = 1 - sonar[i]

        index = self.__numSonarInput

        for i in range(avsonar_input ):

            self.__activityF1[self.__NEWSTATE][i] = av_sonar[i]
            self.__activityF1[self.__NEWSTATE][i + sonar_input] = 1 - av_sonar[i]

        index  +=  self.__numAVSonarInput

        for i in range( self.__numBearingInput):
            self.__activityF1[self.__NEWSTATE][index + i] = 0.0
        self.__activityF1[self.__NEWSTATE][index + bearing] = 1.0
        index  +=  self.__numBearingInput

        for i in range( range_input):

            self.__activityF1[self.__NEWSTATE][index + i] = target_range
            self.__activityF1[self.__NEWSTATE][index + i + range_input] = 1 - target_range

    def computeChoice (self, Type = 0, numSpace = 0):  #Type = 0是FUZZART Type = 1是ART2,计算出F2层所有节点响应的选择函数，存入activityF2中
        top = 0.0
        bottom = 0.0
        #Predicting numSpace的取值范围 1 - 3
        if (Type == self.__FUZZYART):   #self.__FUZZYART = 0
            for j in range(self.__numCode+1):
                self.__activityF2[j] = 0.0 #self.__activityF2 = (针对输入层F1[][]，F2层所有结点的给出响应后的选择函数T)
                for k in range(numSpace):  #Code activation，k:0 -  > 3 k从0到numSpace
         #            if (self.__gamma[k] > 0.0)
                    top = 0 #选择函数T的分子
                    bottom = self.__alpha[k] #选择函数T的分母
                    for i in range(self.__numInput[k]):
                        top  +=  min (self.__activityF1[k][i],self.__weight[j][k][i])  #fuzzy AND operation
                        bottom  +=  self.__weight[j][k][i]
 #                    self.__activityF2[j] * =  (top/bottom)   # product rule, does not work
                    self.__activityF2[j]  +=  self.__gamma[k]*(top/bottom)  #Code activation

 #              println( "F[" + j + "] = "  +  self.__activityF2[j] )

        elif (Type == self.__ART2):   #self.__ART2 = 1
            for j in range(self.__numCode+1):
                self.__activityF2[j] = 0.0
                for k in range(numSpace):
                    top = 0
                    for i in range(self.__numInput[k]):
                        top  +=  self.__activityF1[k][i]*self.__weight[j][k][i]
                    top /=  self.__numInput[k]
                    self.__activityF2[j]  +=  top
        else:
            pass

    def doChoice(self):  #选出F2中响应函数最大的节点c
        max_act =  -1.0
        c =  -1

        for j in range(self.__numCode+1):
            if (self.__activityF2[j] > max_act):
                max_act = self.__activityF2[j]
                c = j
        return (c)


    def isNull (self, x = [], n = 0):  #用来判断[]x是否全部为0
        for i in range(n):
            if (x[i]!= 0):
                return (False)
        return (True)


    def doMatch (self, k = 0, j = 0):    #Learning：Template matching,对第k个输入Field作匹配，依次产生m^j^c1, m^j^c2,m^j^c3

        m = 0.0
        denominator = 0.0  #分母

        if (self.isNull(self.__activityF1[k],self.__numInput[k])): #如果Xck全部为0，那就返回匹配函数Mjck = 1，这样做是为了防止分母为0
            return (1.0)

        for i in range(self.__numInput[k]):
            m  +=  min (self.__activityF1[k][i], self.__weight[j][k][i])  #fuzzy AND operation
            denominator  +=  self.__activityF1[k][i]

 #      println ("Code " + j +  " match " + m/denominator)
        if (denominator == 0):#再次防止分母为0
            return (1.0)
        return (m / denominator)


    def doComplete (self,j = 0, k = 0):  # self.__activityF1[k] <  <  ====== self.__weight[j][k]
        for i in range(self.__numInput[k]):
            self.__activityF1[k][i] = self.__weight[j][k][i]


    def doInhibit (self, j = 0, k = 0):  #inhibit 抑制?
        for i in range(self.__numInput[k]):
            if (self.__weight[j][k][i] == 1):
                self.__activityF1[k][i] = 0


    def doSelect (self, k = 0):  #选择ACtion Field 中获胜的动作，把最大可能的动作置为1, 其余的置为0,返回获胜者 winner
        winner = 0
        max_act = 0

        for i in range(self.__numInput[k]):
            if (self.__activityF1[k][i] > max_act):
                max_act = self.__activityF1[k][i]
                winner = i

        for i in range(self.__numInput[k]):
            self.__activityF1[k][i] = 0
        self.__activityF1[k][winner] = 1
        return(winner)

    def doLearn(self, J, type):  #对最相似的节点J进行学习
        rate = 0.0 #学习率

        if ((not self.__newCode[J]) or self.__numCode < self.__capacity):  #如果Jcommitted node or J 为 Uncommitted node 但是 当前容量还足够开拓新结点

            if (self.__newCode[J]):
                rate = 1 #如果是新节点，那么就快速学习，学习率为1 ,即 Wj - ck = X - ck
            else :
                rate = self.__beta  #Math.abs(r - reward)β = 1

            for k in range(self.__numSpace):
                for i in range(self.__numInput[k]):
                    if (type == self.__FUZZYART):
                        self.__weight[J][k][i] = (1 - rate) * self.__weight[J][k][i]  + rate*min(self.__weight[J][k][i],self.__activityF1[k][i])
                    elif (type == self.__ART2):
                        self.__weight[J][k][i] = (1 - rate) * self.__weight[J][k][i]  + rate*self.__activityF1[k][i]

            if (self.__newCode[J]):
                self.__newCode[J] = False
                self.createNewCode ()


    def doOverwrite(self, J):  #重写函数，对于J结点 w[j][k][i]  <  <  =====  self.__activityF1[k][i]
        for k in range(self.__numSpace):
            for i in range(self.__numInput[k]):
                self.__weight[J][k][i] = self.__activityF1[k][i]


    def displayActivity(self, k):  #输出ActivityF1[k]
        print ("Space " , k, " : ")
        for i in range(self.__numInput[k]-1):
            print ("%.2f"%(self.__activityF1[k][i]) + ", ", end = '')
        print("")


    # def displayActivity2(self, PrintWriter pw, k = 0):  #把ActivityF1[k]写入文件中
# 这个函数暂时没有用
    #     pw.print ( "AV"  +  self.__agentID  +  " Space " + k + " : " )
    #     for (int i = 0 i < self.__numInput[k] -1 i +=  1)
    #         pw.print ("%.2f"%(self.__activityF1[k][i]) + ", ", end = '')
    #     pw.println ("%.2f"%(self.__activityF1[k][self.__numInput[k] -1]))


    def displayVector(self, s = "", x = [], n = 0):  #输出向量[]x
        print (s +  " : ")
        for i in range(n-1):
            print ("%.2f"%(x[i]) + ", ", end = '')
        print("")

    def displayState (self,s = "", x = [], n = 0):  #输出状态State (10 + 8)
        print (s +  "   Sonar: [", end = '')
        index = 0
        for i in range(self.__numSonarInput): #输出10个Sonar输入
            print ("%.2f"%(x[index + i]) + ", ", end = '')
        print ("%.2f"%(x[index + self.__numSonarInput -1]), end = '')

        print("]", end = '')
        print("TargetBearing: [", end = '')
        index = self.__numSonarInput
        for i in range(self.__numBearingInput):
            print ("%.2f"%(x[index + i]) + ", ", end = '')
        print("%.2f"%(x[index + self.__numBearingInput -1])  +  "]", end = '')

    def doSearchQValue(self, mode = 0, type = 0):  #according to state and action to map reward
        reset = True
        perfectMismatch = False
        QValue = 0.0
        rho = [0.0]*4  #ρ[4] 存放警戒参数
        match = [0.0] * 4  #match[4]存放四个匹配函数

        if (mode == self.__INSERT): #三种警戒参数 self.__INSERT = 0
            for k in range(self.__numSpace):
                rho[k] = 1  # 1 1 1 1
        elif (mode == self.__LEARN):  # self.__LEARN = 1
            for k in range(self.__numSpace):
                rho[k] = self.__b_rho[k]  # 0.2 0.2 0.5 0
        elif (mode == self.__PERFORM):  #predict阶段用不着警戒参数，Learning阶段才用 self.__PERFORM = 2
            for k in range(self.__numSpace):
                rho[k] = self.__p_rho[k]  # 0 0 0 0
        else:
            pass

 #        println ("Running searchQValue:")
        self.computeChoice(type,2)  #map from state action to reward or or self.__numSpace = 2 代表 reward

        while (reset and not perfectMismatch):
            reset = False
            self.__J = self.doChoice ()  #Code competition,函数返回获胜的结点即T值最大的点的Index j
            for k in range(self.__numSpace):
                match[k] = self.doMatch(k,self.__J)     #Learning：Template matching, 把3 个 Field 的匹配函数 m 算出来存入 match[]中
            if (match[self.__CURSTATE] < rho[self.__CURSTATE] or match[self.__ACTION] < rho[self.__ACTION] or match[self.__REWARD] < rho[self.__REWARD]):  #如果三者之中有一个不满足警戒参数
                if (match[self.__CURSTATE] == 1):  #如果说match[0] = 1 则证明所有的结点都没能匹配上当前的状态，那这就是一个新状态
                    perfectMismatch = True #完美不匹配
                    if (self.__Trace):
                        print("Perfect mismatch. Overwrite code " + str(self.__J))

                else :   #如果3个filed发生了不匹配并且当前获胜的节点J不是uncommitted Node
                    self.__activityF2[self.__J] =  -1.0 #把结点 j 排除在外, rechoose a node 
                    reset = True
# ???
                    for k in range(1):  # raise vigilance of State 对 m^j^c0也就是Filed Curstate 对应的警戒参数提升epilson 
                        if (match[k] > rho[k]):
                            rho[k] = min(match[k] + self.__epilson,1)

        if (mode == self.__PERFORM):   #PERFORM阶段是 F2 层的权值对 F1层的参数产生影响，对应函数doComplate
            self.doComplete (self.__J,self.__REWARD) #把选中结点的权值W对应的Reward 分量赋值给 self.__activityF1[Reward] F2 - >  F1
            if(self.__activityF1[self.__REWARD][0] == self.__activityF1[self.__REWARD][1] and self.__activityF1[self.__REWARD][0] == 1): #initialize Q value，如果 选中的结点是 Uncommitted node 就会满足这个条件
                if (self.__INTERFLAG):
                    QValue =  AGENT.initialQ #initialQ = 0.5
                else :
                    QValue =  AGENT.initialQ
            else :
                QValue = self.__activityF1[self.__REWARD][0]

        elif (mode == self.__LEARN):   #LEARN阶段是 F1 层的权值对 F2 层的参数产生影响，对应函数doOverWrite
            if (not perfectMismatch):
                self.doLearn(self.__J,type)  #如果选中的节点J不是新节点就对节点J的权重进行学习
            else :
                self.doOverwrite (self.__J)  #如果选中的节点J是新节点就直接把 F1 层的权重 - >  F2层的新节点J
        return (QValue)

    def getMaxQValue (self,method, train, maze):  #输入参数为 Maze - 地图类 method -  Q(0) or Sarsa(1)  train - 是否是训练模式
      #函数作用应该是求出最大化的Q值
        QLEARNING = 0  #Q - learning 和 sarsa 两种算法
        #SARSA = 1
        Q = 0.0

        if(maze.isHitMine( self.__agentID ) ):   #elif self.__currentBearing == hit mine agentID是Agent编号，每一个Agent都蕴含着一个FALCON网络
            Q = 0.0
        elif( maze.isHitTarget( self.__agentID ) ):
            Q = 1.0  #elif self.__currentBearing == reach target
        else :
            if(method == QLEARNING):  #q learning TDerr = r  +  γmaxQ'(s',a') - Q(s,a) 因此需要根据当前状态求出下一状态和动作的最大Q值
                for i in range(self.__numAction):  #self.__numAction = 5 根据当前状态S2求出所有可选动作中Q值最大的动作，此时还处在状态S2还不知道真实地A2选择了哪个动作
                    self.setAction(i)  #依次把F1层 Action Field 设置为 五个动作，查看每个动作能获得的Q值
                    tmp_Q = self.doSearchQValue(self.__PERFORM,self.__FUZZYART)  # self.__PERFORM 阶段是 F2对F1层产生影响，doSearchQvalue函数 根据 state and action 确定 reward
                    if(tmp_Q > Q):
                         Q = tmp_Q
            else :                                 #sarsa 根据state来选择节点J从而获得 J节点的Action
                next_a = self.doSelectAction( train, maze )   # 这一步做的是仅仅根据状态state选择动作 S1 - >  A1 （1 - ε）概率选择Q最大的动作，ε概率随机选择动作
                self.setAction(next_a)   # set action
                Q = self.doSearchQValue(self.__PERFORM,self.__FUZZYART)  #doSearchQValue 根据state和action 来选择节点J从而获得 J节点的Reward
        return Q

    def doSearchAction(self, mode = 0, type = 0):   #根据 state 来映射动作
        reset = True
        perfectMismatch = False
        action = 0
        rho   = [0.0] * 4
        match =[0.0]* 4

        if (mode == self.__INSERT):
            for k in range(self.__numSpace):
                rho[k] = 1
        elif (mode == self.__LEARN):
            for k in range(self.__numSpace):
                rho[k] = self.__b_rho[k]
        elif (mode == self.__PERFORM):
            for k in range(self.__numSpace):
                rho[k] = self.__p_rho[k]

        print ("Running searchAction")
        if (self.__Trace):
            print("\n Input activities:")
            self.displayState("STATE", self.__activityF1[self.__CURSTATE], self.__numInput[self.__CURSTATE])
            self.displayActivity(self.__ACTION)
            self.displayActivity(self.__REWARD)


        # if (mode == self.__LEARN):
#             if (self.__Trace) println ("reward = " + r +  " prev = " + reward)
#             if (r == 0 or r <  = reward) 
#                   if (self.__Trace) println ("reset action")
#                   for i in range(self.__numInput[self.__ACTION]) :
                    # self.__activityF1[self.__ACTION][i] = 1 - self.__activityF1[self.__ACTION][i]

#               reward = r

        self.computeChoice(type,1)   #1 - choice function is computed based on state only 
                                  #3 - choice function is computed based on state, action, and value 
        while (reset and not perfectMismatch):
            reset = False
            self.__J = self.doChoice ()
            for k in range(self.__numSpace):
                match[k] = self.doMatch(k,self.__J)
            if (self.__Trace):
                print("winner = " + self.__J)
                self.displayState ("self.__weight[self.__J][STATE] ",self.__weight[self.__J][self.__CURSTATE],self.__numInput[self.__CURSTATE])
                self.displayVector ("self.__weight[self.__J][self.__ACTION]",self.__weight[self.__J][self.__ACTION],self.__numInput[self.__ACTION])
                self.displayVector ("self.__weight[self.__J][self.__REWARD]",self.__weight[self.__J][self.__REWARD],self.__numInput[self.__REWARD])
                print ("Winner " + self.__J + " act "  +  "%.2f"%(self.__activityF2[self.__J])  +  " match[State] = "   +  "%.2f"%(match[self.__CURSTATE])  + " match[action] = "  +  "%.2f"%(match[self.__ACTION])  + " match[reward] = "  +  "%.2f"%(match[self.__REWARD]))

             # Checking match in all channels
            if (match[self.__CURSTATE] < rho[self.__CURSTATE] or match[self.__ACTION] < rho[self.__ACTION] or match[self.__REWARD] < rho[self.__REWARD]):
                if (match[self.__CURSTATE] == 1):
                    perfectMismatch = True
                    if (self.__Trace):
                         print ("Perfect mismatch. Overwrite code " + self.__J)

                else :
                    self.__activityF2[self.__J] = -1.0
                    if (self.__Trace):
                        print("Reset Winner " + self.__J + " rho[State] " + rho[self.__CURSTATE] + " rho[Action] " + rho[self.__ACTION] + " rho[Reward] " + rho[self.__REWARD])                
                    reset = True

                    for k in range(1):  # raise vigilance of State only
                        if (match[k] > rho[k]):
                            rho[k] = min (match[k] + self.__epilson,1)


        if (mode == self.__PERFORM):
            if (self.__newCode[self.__J]):
                action =   -1
            else :
                self.doComplete (self.__J,self.__ACTION) #把选中结点的权值W对应的Action 分量赋值给 self.__activityF1[Action] F2 - >  F1
                action = self.doSelect (self.__ACTION)  #选择ActivityF1[Action]节点中最大的动作，并且获胜的动作取全部1，其余的均置为 0 

        elif (mode == self.__LEARN):
            if (not perfectMismatch):
                self.doLearn(self.__J,type)
            else :
                self.doOverwrite (self.__J)
        else:
            pass
        return (action)

    def loop_path(self):  #判断路径是否成环
        k = 0
        for k in range(self.__step -1, -1, -1): #step是agent当前已经走了多少步
            if( ( self.__current[0] ==  self.__path[k][0] ) and ( self.__current[1] ==  self.__path[k][1] ) ):  #当前的坐标是否和之前的坐标重合
                return( k )
        return( -1 )


    def get_except_action(self):  #如果当前的Agent成环转圈圈了，求出发生回路点的下一步所选择的动作

        rep_step = 0
        a = 0

        new_pos = [0] * 2
        rep_step = self.loop_path()  #记录下和当前节点成环的Path之前走过的节点
        if( rep_step  <  0 ):
            return(  -1 )
        for a in range(self.__numAction):  # 这一步应该是求出环路的交界点rep_Step的下一步所选择的动作

            self.virtual_move( a - 2, new_pos )  #new_pos变成了current按照a动作移动后的位置
            if( ( new_pos[0] ==  self.__path[rep_step + 1][0] ) and ( new_pos[1] ==  self.__path[rep_step + 1][1] ) ):
                return( a )
        return( -1 )


    def doSelectAction (self, train, maze): # 根据ε - greedy选择一个最可能的动作 sarsa算法中选择下一个s'的a' 输入参数为 train - 是否是训练 maze - 地图类
        #self.__numAction = 5
        qValues = [ 0.0 ] * (self.__numAction)
        selectedAction =  -1


         #get qValues for all available actions
        for i in range(self.__numAction):
            self.setAction(i)
            qValues[i] = self.doSearchQValue(self.__PERFORM,self.__FUZZYART)


        maxQ =  float("-inf")  #最小值
        doubleValues = [0]*len(qValues)
        maxDV = 0  #代表有多个动作的Q值相同
 #  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  > 2020.2.3 >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  > 
         #Explore
        if ( random.random()  <  self.QEpsilon and train == True ):   #如果刚好随机到ε内 直接随机选择
            selectedAction =  -1

        else :   # ε - greddy

            for action in range(len(qValues)):#从所有备选动作中选择Q值最大的动作
                # if(maze.nextReward(action - 2) > 0.5)  #add in rules
                #     selectedAction = action
                #     maxDV = 0
                #     break

                if( qValues[action]  >  maxQ ):
                    selectedAction = action
                    maxQ = qValues[action]
                    maxDV = 0
                    doubleValues[maxDV] = selectedAction

                elif( qValues[action] ==  maxQ ):
                    maxDV +=  1
                    doubleValues[maxDV] = action

            if( maxDV  >  0 ):   #多个动作Q值相同且都为最大则随机选一个
                randomIndex = random.randint(0, maxDV)
                selectedAction = doubleValues[ randomIndex ]

         # Select random action if all qValues ==  0 or exploring.
        if ( selectedAction ==   -1 ):
            if(self.__Trace):
                print("random action selected!")

            selectedAction = random.randint(0, len(qValues) - 1)  #如果是探索操作则随便选择一个动作

        return selectedAction

    def doSelectValidAction( self, train, maze ):  #选择有效操作
        qValues = [0.0] * self.__numAction
        selectedAction =  -1
        #except_action  =  -1
        #k = 0
        
 #        if (self.__detect_loop)           
 #        except_action = get_except_action()

         #get qValues for all available actions

        validActions = [0] * len(qValues)
        maxVA = 0

        for i in range(self.__numAction): 
            if (maze.withinField (self.__agentID, i - 2) == False):  #如果agent跑到边界外面
                qValues[i] =  -1.0
 #                  println ( "action "  +  i  +  " invalid")
            else :    #agent没有到边界外，在动作i对应的方向上还能继续移动
                self.setAction( i )      #设置动作
                qValues[i] = self.doSearchQValue( self.__PERFORM, self.__FUZZYART )    #计算q值
                validActions[maxVA] = i
                maxVA +=  1

        if (maxVA == 0):

 #             println ( "self.__current = (" + self.__current[0] + "," + self.__current[1] + ")  Bearing = "  +  self.__currentBearing)
 #             println ( "*** No valid action *** ")

             return ( -1)

             # Explore
        if( random.random()  <  self.QEpsilon and train == True):
         # Select random action if all qValues ==  0 or exploring.
            if(self.__Trace):
                print("random action selected!")
            randomIndex = random.randint(0, maxVA - 1)   #如果是探索操作，则在可以选择的动作maxVA里随机选择一个
            selectedAction = validActions[randomIndex]

        else :
            maxQ =  float("-inf")
            doubleValues =[0]*len(qValues)
            maxDV = 0

            for vAction in range(maxVA):

                action = validActions[vAction]
                #   print ( "action["  +  action  +  "] = "  +  qValues[action])
                #    if (self.__detect_loop)
                #     if( except_action ==  action )
                #         continue

 #               println ( "   nextReward["  +  action  +  "] = "  +  maze.nextReward (self.__agentID, action - 2))
                # if (self.__look_ahead):
                #        if( maze.nextReward (self.__agentID, action - 2)  >  0.5):   #add in rules 
                #         selectedAction = action
                #         maxDV = 0
                #         break

                if( qValues[action]  >  maxQ ):
                    selectedAction = action
                    maxQ = qValues[action]
                    doubleValues[maxDV] = selectedAction
                    maxDV = 1
                elif( qValues[action] ==  maxQ ):
                    doubleValues[maxDV] = action
                    maxDV +=  1


            if( maxDV  >  1 ):     # more than 1 with max value
                randomIndex =  random.randint(0, maxDV - 1) 
                selectedAction = doubleValues[ randomIndex ]

 #            print( "Best valid action is "  +  selectedAction  +  " with maxQ  = "  +  maxQ)

        if (selectedAction ==  -1):
               print ( "No action selected")

        return selectedAction

     # Direct Access
    def doDirectAccessAction (self, agt, train, maze):

        selectedAction  = 0 # from 0 to 4

        if (agt!= self.__agentID):
            print ( "ID not consistent")

         # first try to select an action
         #setState (maze.getSonar(), (maze.getTargetBearing() - maze.getCurrentBearing() + 10)%8)
        self.initAction()    # initialize action to all 1's
        self.setReward (1)   # search for actions with good reward

# no close match，只能创建一个新节点
        selectedAction = self.doSearchAction (self.__PERFORM, self.__FUZZYART)
        if (random.random()  <  self.QEpsilon or selectedAction == -1 or  maze.withinField (agt, selectedAction - 2) == False): # not valid action，或者这个选择的新动作无法执行，否则会跑出地图外
            if (self.__Trace):
                print("random action selected!")
            validActions = [0]*self.__numAction
            maxVA = 0

 #            print ("Valid actions :")
            for i in range(self.__numAction):
                if (maze.withinField (agt, i - 2)):     # valid action
                    # print (" "  +  i)
                    validActions[maxVA] = i
                    maxVA +=  1

 #            println (" ")
            if (maxVA > 0):
                randomIndex = random.randint(0 , maxVA - 1)
                selectedAction = validActions[randomIndex]
            else :
                selectedAction =  -1

   #         else :
   #             println ( "Chosen valid action is "  +  selectedAction


 #        if (selectedAction ==  -1)
 #               println ( "No action selected")
 #           if (maze.withinField (agt, selectedAction - 2) == False)
 #              println("WARNING: selectedaction "  +  selectedAction  +  " out of field")

        return selectedAction

    def findKMax (self, v, n, K):   #寻找v中前k个最大值
        temp = 0
        tempf = 0.0
        maxIndex = [0]* K
        index = [0]* K

        for i in range(n):
            index[i] = i

        for k in range(K):
            for i in range(n - 1, k, -1):
                if (v[i -1] < v[i]):
                    tempf = v[i]
                    v[i] = v[i -1]
                    v[i -1] = tempf
                    temp = index[i]
                    index[i] = index[i -1]
                    index[i -1] = temp

            maxIndex[k] = index[k]
        return(maxIndex)


    def doCompleteKMax ( self, k_max = [0]):   #k_max数组里存了F2层中数值最大的3个Node的下标
        actualK = 0
        j = 0
        predict = False

        if (self.__numCode < self.__KMax):     #self.__KMax = 3
            actualK = self.__numCode
        else :
            actualK = self.__KMax

        for i in range(self.__numInput[self.__ACTION]) :   #五个动作ACtion，对于每个动作 i 如果 F2层的T函数的值大于0.9 就对F1层的Actin Field做更新
            self.__activityF1[self.__ACTION][i] = 0
            for k in range(actualK):  #k_max数组中的值
                j = k_max[k]
                if (self.__activityF2[j] > 0.9):      # self.__threshold of activity for predicting
                    #good move  #bad move
                    self.__activityF1[self.__ACTION][i]  +=  self.__activityF2[j] * (self.__weight[j][self.__REWARD][0]*self.__weight[j][self.__ACTION][i] - 
                    (1 - self.__weight[j][self.__REWARD][0])*self.__weight[j][self.__ACTION][i])
                    predict = True

        return( predict )


    def doSelectDualAction(self,type = 0):   #双重(Dual)动作
        #reset = True
        action = 0
        rho = [0.0]*4
        k_max=  []

        for k in range(self.__numSpace):
            rho[k] = self.__p_rho[k]  #0 0 0 0 

        if (self.__Trace):
            print ("Input activities")
            self.displayActivity(self.__CURSTATE)
            self.displayActivity(self.__ACTION)
            self.displayActivity(self.__REWARD)

        self.computeChoice(type,1)
 #      for j in range(self.__numCode):
 #          println ("F2[" + j + "] =  " + self.__activityF2[j])

        if (self.__numCode > self.__KMax):
            k_max = self.findKMax (self.__activityF2, self.__numCode, self.__KMax) #从F2层中选择T函数值最大的KMax(3)个Node的下标
 #          for (int j = 0 j < K j +=  1)
 #              println ("k_max[" + j + "] =  " + k_max[j])
            if (not self.doCompleteKMax (k_max) ):  #如果该函数返回的是False
                self.__J = self.doChoice()  #Code competition,函数返回获胜的结点即T值最大的点的Index j
        else :
            self.__J = self.doChoice()

        action = self.doSelect (self.__ACTION)
        return (action)


    def virtual_move( self, a,  res ):   #虚拟移动，[]res传入的是new int[2] a是传入的 a - 2 a∈[0, 4] a - 2 就是做了下转化
     # 函数作用对当前的坐标current 和 当前朝向 bearing 计算经过 动作a后的朝向 并且 按当前方向移动一个单位            
        bearing = ( self.__currentBearing  +  a  +  8 ) % 8

        res[0] = self.__current[0]
        res[1] = self.__current[1]

        if bearing == 0:
                res[1] -=  1
        elif  bearing == 1:
                res[0] +=  1
                res[1] -=  1
        elif  bearing ==  2:
                res[0] +=  1
        elif  bearing ==  3:
                res[0] +=  1
                res[1] +=  1
        elif  bearing ==  4:
                res[1] +=  1
        elif  bearing ==  5:
                res[0] -=  1
                res[1] +=  1
        elif  bearing ==  6:
                res[0] -=  1
        elif  bearing ==  7:
                res[0] -=  1
                res[1] -=  1
        else:
            pass

        return


    def turn(self, d = 0 ):

        self.__currentBearing = ( self.__currentBearing  +  d  +  8 ) % 8


    def move( self, a = 0,  succ = False ):

        if self.__step >= self.__max_step:
            print(str(self.__agentID) + "已到达最大步数")
            return
        self.__currentBearing = ( self.__currentBearing  +  a  +  8 ) % 8
        self.__step += 1
        if( not succ ):
            self.__path[self.__step][0] = self.__current[0]
            self.__path[self.__step][1] = self.__current[1]
            return


        if self.__currentBearing == 0:
                self.__current[1] -=  1

        elif self.__currentBearing == 1:
                self.__current[0] +=  1
                self.__current[1] -=  1

        elif self.__currentBearing == 2:
                self.__current[0] +=  1

        elif self.__currentBearing == 3:
                self.__current[0] +=  1
                self.__current[1] +=  1

        elif self.__currentBearing == 4:
                self.__current[1] +=  1

        elif self.__currentBearing == 5:
                self.__current[0] -=  1
                self.__current[1] +=  1

        elif self.__currentBearing == 6:
                self.__current[0] -=  1

        elif self.__currentBearing == 7:
                self.__current[0] -=  1
                self.__current[1] -=  1

        else:
            pass


        self.__path[self.__step][0] = self.__current[0]
        self.__path[self.__step][1] = self.__current[1]
        return


    def get_J(self):

        return(self.__J )


      # dummy methods required by abstract AGENT class
    def doLearnACN (self):
        pass
    def setprev_J (self):
        pass
    def computeJ (self, maze):
        return 0.0
    def setNextJ (self, j = 0.0):
        pass
