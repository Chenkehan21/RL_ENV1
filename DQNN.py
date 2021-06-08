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


class DQN(AGENT):
    NUM_LAYERS = 3
    N = 18
    H = 36
    M = 1

    def __init__ (self, av_num) :
        super(DQN,self).__init__()
        print("DQN init")
        self.numSpace=4  # 0-State 1-Action 2-Reward 3-New State 
        self.numSonarInput=5
        self.numAVSonarInput=0 
        self.numBearingInput=8
        self.numRangeInput=0
        self.numAction=5
        self.numReward=2
        self.complementCoding=1

        self.BATCH_SIZE = 128
        self.LR = 0.00024
        self.GAMMA = 0.8
        self.EPISILO_START = 0.9
        self.EPISILO_END = 0.05
        self.EPISILO = 0.9
        self.EPISILO_DECAY = 200
        self.MEMORY_CAPACITY = 10000
        self.Q_NETWORK_ITERATION = 100
        self.ENV_A_SHAPE = 0
          
        self.PERFORM =0
        self.LEARN   =1
        self.__agent_num = av_num
        self.__preReward = 0.0
        self.__Trace = False
        #self.detect_loop = False
        #self.look_ahead = False
        #self.__end_state
        self.__current = [0, 0]
        self.__max_step = 0
        self.__step = 0
        self.__currentBearing = 0
        self.__path = []

        self.MinTestError = 999999.99
        self.train_flag = True
        
        self.input_dim = self.numSonarInput*2+self.numAVSonarInput+self.numBearingInput
        self.__numState = self.input_dim
        self.output_dim = self.numAction
        self.Input = np.zeros(self.input_dim, dtype=float)
        self.action = np.zeros(self.numAction, dtype=float)
        self.Output = np.zeros(self.numAction, dtype=float)
        self.Target = np.zeros(self.numAction, dtype=float)
        
        self.GenerateNetwork()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.input_dim * 2 + 2))

    def GenerateNetwork(self):
        self.eval_net = Net(self.input_dim, self.output_dim)
        self.target_net = Net(self.input_dim, self.output_dim)
        self.eval_net.zero_grad()
        self.target_net.zero_grad()
        self.optimizer = optim.RMSprop(self.eval_net.parameters())
        # self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.LR)
        # self.loss_func = nn.MSELoss() 

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))

        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.learn_step_counter % self.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # return weight && bias
        self.learn_step_counter+=1
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.__numState])
        batch_action = torch.LongTensor(batch_memory[:, self.__numState:self.__numState+1].astype(int)) #astype array改变类型
        batch_reward = torch.FloatTensor(batch_memory[:, self.__numState+1:self.__numState+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-self.__numState:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action) #gather(input, dim, index, out=None, sparse_grad=False) → Tensor
        q_next = self.target_net(batch_next_state).detach() #根据状态s'选择a'tar_net下两个动作的Q值
        q_target = batch_reward + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        # loss = self.loss_func(q_eval, q_target)
        loss = F.smooth_l1_loss(q_eval, q_target)

        self.optimizer.zero_grad()#d_weights = [0] * n 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



    def SimulateNet(self, input_value, output_value, target_value):
        pass
        # x = torch.tensor(input_value, dtype=torch.float32)
        # y = torch.tensor(target_value, dtype=torch.float32)
        # if training == True:
        #     self.optimizer.zero_grad()
        #     output = self.net(x)
        #     loss = self.criterion(output, y)
        #     loss.backward()
        #     self.optimizer.step()
        # else:
        #     output = self.net(x)
        # self.Output[0] = output.item()
    


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
            self.QGamma =  0.5
        else :
            self.QGamma =  0.9
    
    def getState(self, sonar, av_sonar, bearing, r):
        State = np.zeros(self.input_dim, dtype=float)
        for i in range(self.numSonarInput):
            State[i] = sonar[i]
            State[i+self.numSonarInput] = 1-sonar[i]
        
        index  = self.numSonarInput*2
        for i in range(self.numAVSonarInput):
            State[index+i] = av_sonar[i]

        index += self.numAVSonarInput
        for i in range(self.numBearingInput):
            State[index+i] = 0.0
        State[index+bearing] = r
        return State

    def setState(self, sonar, av_sonar, bearing, r):
        
        for i in range(self.numSonarInput):
            self.Input[i] = sonar[i]
            self.Input[i+self.numSonarInput] = 1-sonar[i]
        index  = self.numSonarInput*2
        for i in range(self.numAVSonarInput):
            self.Input[index+i] = av_sonar[i]

        index += self.numAVSonarInput
        for i in range(self.numBearingInput):
            self.Input[index+i] = 0.0
        self.Input[index+bearing] = r

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
        index = self.numSonarInput*2 + self.numAVSonarInput + self.numBearingInput
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
            self.Input[i+self.numSonarInput] = 1-sonar[i]
        index = self.numSonarInput*2
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
    
    def doSelectValidAction(self, state, maze):
        flag = False
        for i in range(self.numAction):
            if maze.withinField(self.__agent_num, i-2) == True:
                flag = True
                break
        
        if flag == False:
            return -1

        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= self.EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            # print(action_value)
            for i in range(self.numAction):
                if maze.withinField(self.__agent_num, i-2) == False:
                    action_value[0][i] = -500
            action = torch.max(action_value, 1)[1].data.numpy() #max(input, dim) -> return[input中的最大值, 最大值的下标]
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
            
        else: # random policy
            action = np.random.randint(0,self.numAction)
            while (maze.withinField(self.__agent_num, action-2) == False):
                action = np.random.randint(0,self.numAction)
            action = action if self.ENV_A_SHAPE ==0 else action.reshape(self.ENV_A_SHAPE)


        return action

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
            # print(str(self.__agentID) + "已到达最大步数")
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
        pass
        # Qvalue = 0.0
        # if mode == self.LEARN:
        #     self.SimulateNet(self.Input, self.Output, self.Target, True)
        # elif mode == self.PERFORM:
        #     self.SimulateNet(self.Input, self.Output, self.Target, False)
        # Qvalue = self.Output[0]
        # return Qvalue

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