######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# Cost Function basis
######
import numpy as np

class Cost:
    indices    = ['Cost']
    weight     = [1]
    uncertaintyStandardDeviation = [0]
    name       = 'Cost'
    costIDX0   = []

    doSimulation = True # false if cost is based on stoveVectorData

    def __len__(self):
        return len(self.indices)

    def getStoreVectorData(self):
        return []

    def getIDX0(self,patient,circadapt):
        if self.costIDX0==[]:
            return 0
        else:
            return self.costIDX0.getIDX0(patient,circadapt)

    def getCostSettings(self):
        return []

    def getName(self,idx):
        return 'Cost'

    def getCostMeas(self,patient):
        return np.array([0])

    def getCostMod(self,patient):
        return np.array([0])

    def getCost(self,patient,model_instance):
        return [0]
    def getCost2(self,patient,model_instance):
        return [0]

    def save(self,filename):
        data = self.getSaveData()
        np.save(filename,data,allow_pickle=True)

    def getSaveData(self):
        if self.costIDX0 == []:
            saveCostIDX0 = []
        else:
            saveCostIDX0 = self.costIDX0.getSaveData()
        return {'indices':self.indices,'weight':self.weight,'name':self.name,'costIDX0':saveCostIDX0,'uncertaintyStandardDeviation':self.uncertaintyStandardDeviation}

    def load(self,filename):
        data = np.load(filename,allow_pickle=True)
        self.setSaveData(data.item())

    def setSaveData(self,data):
        self.indices = data['indices']
        self.weight = data['weight']
        self.uncertaintyStandardDeviation = data['uncertaintyStandardDeviation']
        self.name = data['name']
        self.costIDX0 = []
        if data['costIDX0']!=[]:
            pass # CHange this

from .CostStrain import CostStrain
from .CostStrain2 import CostStrain2
from .CostStrain3 import CostStrain3
from .CostStrain4 import CostStrain4
from .CostStrain5 import CostStrain5
from .CostStrain6 import CostStrain6
from .CostStrain2vol1 import CostStrain2vol1
from .CostStrain2vol2 import CostStrain2vol2
from .CostStrainIndices import CostStrainIndices
