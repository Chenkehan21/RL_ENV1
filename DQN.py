import numpy as np
import random
from AGENT import AGENT
from Maze import Maze
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data.normal_(0,0.1)
        self.head = nn.Linear(64, output_dim)
        self.head.weight.data.normal_(0,0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

class myloss(nn.Module):
    def __init__(self, gain):
        super(myloss, self).__init__()
        self.gain = gain
    
    def forward(self, output_value, target_value):
        err = target_value-output_value
        err = self.gain * output_value * (1-output_value) * err
        loss = 0.5 * (err**2)
        return loss

class DQN(AGENT):
    

    def __init__ (self, av_num) :
        super(DQN,self).__init__()
        print("BP init")
        self.numSpace=4  # 0-State 1-Action 2-Reward 3-New State 
        self.numSonarInput=5
        self.numAVSonarInput=0
        self.positionInput = 2
        self.phoInput 
        self.numBearingInput=8
        self.numRangeInput=0 
        self.numAction=5
        self.numReward=2
        self.complementCoding=1

        self.numState = self.numSonarInput + self.positionInput + self.phoInput + self.IDinput

        self.BATCH_SIZE = 128
        self.LR = 0.00024
        self.GAMMA = 0.9
        self.EPISILO = 0.9
        self.MEMORY_CAPACITY = 6000
        self.ENV_A_SHAPE = 0

        self.PERFORM =0
        self.LEARN   =1
        self.INSERT  =2
        self.__agent_num = av_num
        self.__preReward = 0.0
        self.__Trace = False
        #self.detect_loop = False
        #self.look_ahead = False
        #self.__end_state
        self.__current = np.zeros(2, dtype=int)
        self.__max_step = 0
        self.__step = 0
        self.__currentBearing = 0
        self.__path = []

        self.Alpha = 0.5
        self.Eta = 0.25
        self.Gain = 1.0
        self.MinTestError = 999999.99
        self.train_flag = True
        
        
        self.Input = np.zeros(self.numSonarInput+self.numAVSonarInput+self.numBearingInput+self.numAction, dtype=float)
        self.action = np.zeros(self.numAction, dtype=float)
        self.Output = np.zeros(BP.M, dtype=float)
        self.Target = np.zeros(BP.M, dtype=float)
        
        self.GenerateNetwork()

    def GenerateNetwork(self):
        BP.N = self.numSonarInput+self.numAVSonarInput+self.numBearingInput+self.numAction
        self.net = Net(BP.N, BP.H, BP.M)
        self.net.zero_grad()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.criterion = myloss(self.Gain)
    
    def SimulateNet(self, input_value, output_value, target_value, training):
        x = torch.tensor(input_value, dtype=torch.float32)
        y = torch.tensor(target_value, dtype=torch.float32)
        if training == True:
            self.optimizer.zero_grad()
            output = self.net(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
        else:
            output = self.net(x)
        self.Output[0] = output.item()
    


    def doDirectAccessAction(self, agt, train, maze):
        return 0
    def checkAgent(self, outfile):
        pass
    def saveAgent(self, output):
        pass

    def setParameters (self, AVTYPE, immediateReward):
        self.QEpsilonDecay = 0.00001
        self.QEpsilon      = 0.50000

        if (immediateReward): 
            self.QGamma =  0.9
        else :
            self.QGamma =  0.9
            
    def setState(self, sonar, av_sonar, bearing, r):
        
        for i in range(self.numSonarInput):
            self.Input[i] = sonar[i]
        
        index  = self.numSonarInput
        for i in range(self.numAVSonarInput):
            self.Input[index+i] = av_sonar[i]

        index += self.numAVSonarInput
        for i in range(self.numBearingInput):
            self.Input[index+i] = 0.0
        self.Input[index+bearing] = 1.0

    def initAction(self):
        for i in range(self.numAction):
            self.action[i] = 1

    def init_path(self, maxstep,current = [0 , 0]):
        self.__max_step = maxstep
        self.__step = 0
        self.__currentBearing = 0
        self.__path = np.zeros((self.__max_step+1, 2))
        self.__current = current
        

    def resetAction(self):
        for i in range(self.numAction):
            self.action[i] = 1-self.action[i]

    def setAction(self, action):
        index = self.numSonarInput + self.numAVSonarInput + self.numBearingInput
        for i in range(self.numAction):
            self.Input[index+i] = 0
        self.Input[index+action] = 1.0

    def setReward(self, r):
        self.Target[0] = r

    def initReward(self):
        self.Target[0] = 1
    
    def setNewState(self, sonar, av_sonar, bearing, range):
        for i in range(self.numSonarInput):
            self.Input[i] = sonar[i]
        
        index = self.numAVSonarInput
        for i in range(self.numAVSonarInput):
            self.Input[index+1] = av_sonar[i]

        index += self.numAVSonarInput
        for i in range(self.numBearingInput):
            self.Input[index+i] = 0.0
        self.Input[index+bearing] = 0.0

    def getMaxQValue(self, method, train, maze):
        QLEARNING = 0
        # SARSA = 1
        return_Q = 0.0

        if method == QLEARNING:
            for i in range(self.numAction):
                self.setAction(i)
                tmp_Q = self.doSearchQValue(self.PERFORM, 0)

                if tmp_Q>return_Q:
                    return_Q = tmp_Q
        else:
            next_a = self.doSelectAction(train, maze)
            self.setAction(next_a)
            return_Q = self.doSearchQValue(self.PERFORM, 0)
        if maze.isHitMine(self.__agent_num):
            return 0
        elif maze.isHitTarget(self.__agent_num):
            return_Q = 1
        
        return return_Q
    
    def doSelectValidAction(self, train, maze):
        qValues = np.zeros(self.numAction, dtype=float)
        selectedAction = -1
        except_action = -1

        validActions = np.zeros(self.numAction, dtype=int)
        maxVA = 0
        for i in range(self.numAction):
            if maze.withinField(self.__agent_num, i-2) == False:
                qValues[i] = -1.0
            else:
                self.setAction(i)
                qValues[i] = self.doSearchQValue(self.PERFORM, 0)
                validActions[maxVA] = i
                maxVA += 1
        
        if maxVA == 0:
            return -1
        
        if( np.random.random()  <  self.QEpsilon and train == True):
            if(self.__Trace):
                print("random action selected!")
            randomIndex = random.randint(0, maxVA - 1)  
            selectedAction = validActions[randomIndex]

        else :
            maxQ =  float("-inf")
            doubleValues =np.zeros(len(qValues),dtype=int)
            maxDV = 0

            for vAction in range(maxVA):

                action = validActions[vAction]

                if( qValues[action]  >  maxQ ):
                    selectedAction = action
                    maxQ = qValues[action]
                    maxDV = 0
                    doubleValues[maxDV] = selectedAction
                    
                elif( qValues[action] ==  maxQ ):
                    maxDV +=  1
                    doubleValues[maxDV] = action
                    


            if( maxDV  >  0 ):
                randomIndex =  np.random.randint(0, maxDV + 1) 
                selectedAction = doubleValues[ randomIndex ]

        if (selectedAction ==  -1):
               print ( "No action selected")

        return selectedAction

    def doSelectAction(self, train, maze):
        qValues = np.zeros(self.numAction, dtype=float)
        selectedAction =  -1
        for i in range(self.numAction):
            self.setAction(i)
            qValues[i] = self.doSearchQValue(self.PERFORM, 0)

        maxQ =  float("-inf") 
        doubleValues = np.zeros(len(qValues), dtype=int)
        maxDV = 0 

         #Explore
        if ( np.random.random()  <  self.QEpsilon and train == True ):   
            selectedAction =  -1

        else :   

            for action in range(len(qValues)):
                if( qValues[action]  >  maxQ ):
                    selectedAction = action
                    maxQ = qValues[action]
                    maxDV = 0
                    doubleValues[maxDV] = selectedAction

                elif( qValues[action] ==  maxQ ):
                    maxDV +=  1
                    doubleValues[maxDV] = action

            if( maxDV  >  0 ): 
                randomIndex = np.random.randint(0, maxDV)
                selectedAction = doubleValues[ randomIndex ]

        if ( selectedAction ==   -1 ):
            if(self.__Trace):
                print("random action selected!")

            selectedAction = np.random.randint(0, len(qValues) - 1) 

        return selectedAction

    def doSearchAction(self, mode, type):
        return -1
    
    def virtual_move(self, a, res):
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



    def turn(self, d):
        self.__currentBearing = ( self.__currentBearing  +  d  +  8 ) % 8

    def move(self, a, succ):
        if self.__step >= self.__max_step:
            # print(str(self.__agentID) + "?????????????????????")
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

    def doSearchQValue(self, mode, type):
        Qvalue = 0.0
        if mode == self.LEARN:
            self.SimulateNet(self.Input, self.Output, self.Target, True)
        elif mode == self.PERFORM:
            self.SimulateNet(self.Input, self.Output, self.Target, False)
        Qvalue = self.Output[0]
        return Qvalue

    def setTrace(self, t):
        self.Trace = t
    
    def setPrevReward(self, r):
        self.__preReward = r
    
    def getPrevReward(self):
        return self.__preReward


    def decay(self):
        pass
    def prune(self):
        pass
    def purge(self):
        pass
    def penalize(self):
        pass
    def reinforce(self):
        pass
    
    def getNumCode(self):
        return self.H
    def getCapacity(self):
        return self.H
    def doLearnACN(self):
        pass
    def setprev_J(self):
        pass
    def computeJ(self, maze):
        return 0
    def setNextJ(self, J):
        pass