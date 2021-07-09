######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# Model Data
######

from CircAdapt_CaSaPe import CircAdapt_CaSaPe

class Model:
    PdictRef = []

    def __init__(self,dllFile):
        self.dllFile = dllFile

    def getInstance(self,Pdict=[]):
        if not(Pdict==[]):
            model_instance = CircAdapt_CaSaPe(self.dllFile,Pdict=Pdict)
            return model_instance

        if not(self.PdictRef==[]):
            model_instance = CircAdapt_CaSaPe(self.dllFile,'Pref.npy')
            model_instance.setPdict(self.PdictRef)
            model_instance.run()
            return model_instance
        return CircAdapt_CaSaPe(self.dllFile,'Pref.npy')

    def setPdictRef(self,Pdict):
        self.PdictRef = Pdict
