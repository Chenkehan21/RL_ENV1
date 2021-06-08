from abc import ABCMeta
from abc import abstractmethod

#define abstract class Agent
class AGENT(metaclass=ABCMeta): 
    #class's variables
    QAlpha = 0.5
    QGamma = 0.1
    minQEpsilon = 0.005
    initialQ = 0.5
    QEpsilonDecay = 5e-4
    QEpsilon = 0.5
    direct_access = False
    forgetting = False
    INTERFLAG = False
    Trace = True

    def __init__(self):
        pass
    
    #abstract method
    @abstractmethod
    def saveAgent(self, var1):
        pass
    @abstractmethod
    def checkAgent(self, var1):
        pass
    @abstractmethod
    def setParameters(self, var1, var2):
        pass
    @abstractmethod
    def setAction(self, var1):
        pass
    @abstractmethod
    def initAction(self):
        pass
    @abstractmethod
    def resetAction(self):
        pass
    @abstractmethod
    def setState(self, var1, var2, var3, var4):
        pass
    @abstractmethod
    def setNewState(self, var1, var2, var3, var4):
        pass
    @abstractmethod
    def doSearchAction(self, var1, var2):
        pass
    @abstractmethod
    def doSelectAction(self):
        pass
    @abstractmethod
    def doSelectValidAction(self):
        pass
    @abstractmethod
    def doDirectAccessAction(self):
        pass
    @abstractmethod
    def doLearnACN(self):
        pass
    @abstractmethod
    def setprev_J(self):
        pass
    @abstractmethod
    def computeJ(self):
        pass
    @abstractmethod
    def setNextJ(self):
        pass
    @abstractmethod
    def turn(self):
        pass
    @abstractmethod
    def move(self):
        pass
    @abstractmethod
    def doSearchQValue(self):
        pass
    @abstractmethod
    def getMaxQValue(self):
        pass
    @abstractmethod
    def setReward(self):
        pass
    @abstractmethod
    def setPrevReward(self):
        pass
    @abstractmethod
    def getPrevReward(self):
        pass
    @abstractmethod
    def init_path(self):
        pass
    @abstractmethod
    def setTrace(self):
        pass
    @abstractmethod
    def getNumCode(self):
        pass
    @abstractmethod
    def getCapacity(self):
        pass
    @abstractmethod
    def decay(self):
        pass
    @abstractmethod
    def prune(self):
        pass
    @abstractmethod
    def purge(self):
        pass
    @abstractmethod
    def reinforce(self):
        pass
    @abstractmethod
    def penalize(self):
        pass