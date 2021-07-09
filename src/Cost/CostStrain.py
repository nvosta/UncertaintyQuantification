######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# Cost Function comparing RV3, IVS, and LV strain
######

from Cost import Cost
import numpy as np

class CostStrain(Cost):
    def __init__(self,costIDX0=[]):
        self.name = 'Strain'
        self.costIDX0 = costIDX0
        self.doSimulation = False # false if cost is based on stoveVectorData

    def getStoreVectorData(self):
        return [['','','','Time'],['Rv','Patch','Rv1','Ls'],['Rv','Patch','Rv2','Ls'],['Rv','Patch','Rv3','Ls'],['Sv','Patch','Sv1','Ls'],['Lv','Patch','Lv1','Ls']]

    def measGetStrain(self,patient):
        # Load Measured Strain
        measStrain = np.ndarray((len(patient.RVstrain),5))
        measStrain[:,0:3]=patient.RVstrain
        measStrain[:,3]=patient.SVstrain
        measStrain[:,4]=patient.LVstrain

        # Find RV strain characteristics
        meanRV = measStrain.mean(1)

        return measStrain, meanRV

    def measGetStrainOnset(self,patient):
        if 'measStrainOnset' in patient.computedData and 'measTimeOnset' in patient.computedData:
            return patient.computedData['measStrainOnset'], patient.computedData['measTimeOnset']
        measStrain,meanRV = self.measGetStrain(patient)

        idxMin = meanRV.argmin()
        idxConAt = -int(np.ceil(0.050/patient.RVtime[1]))
        idxCon50 = np.argwhere((range(len(meanRV))<idxMin) & (meanRV<.5*np.min(meanRV)))[0][0]
        idxRel50 = np.argwhere((range(len(meanRV))>idxMin) & (meanRV>.5*np.min(meanRV)))[0][0]

        # Measured strain used for t0
        measStrainOnset = np.concatenate((measStrain[idxConAt:-1,0:3],measStrain[:idxCon50,0:3]))
        measTimeOnset = np.concatenate((patient.RVtime[idxConAt:-1],patient.RVtime[:idxCon50]))

        # store data
        patient.computedData['measStrainOnset'] = measStrainOnset
        patient.computedData['measTimeOnset'] = measTimeOnset

        return measStrainOnset, measTimeOnset

    def getModStrain(self,stretch,iOnset):
        if iOnset==-1:
            iOnset=0
        modStrain = np.concatenate((stretch[iOnset:-1,:], stretch[:(iOnset+1),:]))
        modStrain = (modStrain / modStrain[0,:] -1 ) * 100
        return modStrain

    def getDiscretizedStrain(self,modelTime, modelStrain, measurementTime):
        modStrainDiscretized = np.ndarray((len(measurementTime),3))
        for iSeg in range(3):
            modStrainDiscretized[:,iSeg] = np.interp(measurementTime, modelTime, modelStrain[:,iSeg])
        return modStrainDiscretized


    def getIDX0(self,patient=[], model_instance=[],vectorData=[]):
        '''
            Get idx0 for cost function
                select based on strain
                returns idx0
        '''
        if self.costIDX0==[]:
            measStrainOnset, measTimeOnset = self.measGetStrainOnset(patient)

            measTimeOnsetPlot = measTimeOnset.copy()
            measTimeOnsetPlot[measTimeOnsetPlot>measTimeOnsetPlot[-1]] = measTimeOnsetPlot[measTimeOnsetPlot>measTimeOnsetPlot[-1]] - patient.RVtime[-1]

            # Iterate for multiple onsets
            minError = np.Infinity;
            minErrorIdx = np.nan;

            if model_instance==[]:
                iOnsetStart = np.argmax(vectorData['RvPatchRv1Ls']) - 50
            else:
                iOnsetStart = model_instance.onsetQRS() - 50
            iOnsetEnd = iOnsetStart + 100



            # Init discretized modelled strain
            if model_instance==[]:
                modelTime=vectorData['Time']
            else:
                modelTime = model_instance.getTime()
            RVdt = (patient.RVtime[1]-patient.RVtime[0] )
            maxModelIDX = int(round(modelTime[-1]/RVdt))

            modelTimeDiscretized = np.array(range(maxModelIDX)) * RVdt

            #modStrain0 = model_instance.getStretch(['Rv1','Rv2','Rv3'],0)
            if model_instance==[]:
                modStrain0 =  np.array([vectorData['RvPatchRv1Ls'],vectorData['RvPatchRv2Ls'],vectorData['RvPatchRv3Ls']]).transpose()
                modStrain0=modStrain0/modStrain0[0,:]
            else:
                modStrain0 = model_instance.getComputedData('getIDX0_modStrain0',model_instance.getStretch,['Rv1','Rv2','Rv3'],0)

            for iOnset in range(iOnsetStart,iOnsetEnd):
                #modStrain = self.getModStrain(modStrain0,iOnset)
                if model_instance==[]:
                    modStrain = np.concatenate( ( modStrain0[iOnset:,:], modStrain0[1:iOnset+1,:] ))
                    modStrain = (modStrain / modStrain[0,:] - 1 ) * 100
                    if len(modelTime) != len(modStrain):
                        print(modelTime)
                        print(modStrain)

                    modStrainOnset = self.getDiscretizedStrain(modelTime, modStrain, measTimeOnset)
                else:
                    modStrain = model_instance.getComputedData('getIDX0_modStrain'+str(iOnset),self.getModStrain,modStrain0,iOnset)
                    modStrainOnset = self.getDiscretizedStrain(modelTime, modStrain, measTimeOnset)

                # calculate error and compare

                #plt.plot(measTimeOnsetPlot, modStrainOnset)
                #plt.plot(measTimeOnsetPlot,measStrainOnset[:,0])
                #plt.title(str(iOnset))
                #plt.show()

                curError = np.sum((measStrainOnset - modStrainOnset)**2)
                if curError<minError:
                    minError=curError
                    minErrorIdx = iOnset
            return minErrorIdx
        else:
            return self.costIDX0.getIDX0(patient,model_instance,vectorData)

    def getCost(self,patient=[],model_instance=[],modY=[],measY=[],vectorData=[]):
        return np.sqrt(self.getCost2(patient=patient,model_instance=model_instance,vectorData=vectorData))

    def getCost2(self,patient=[],model_instance=[],vectorData=[]):
        '''
            Get Cost Function based on Strain:
             - Find t0
             - Normalize LV strain
             - Calculate cost

            Returns squared error
        '''
        # Load Measured Strain
        measStrain = np.ndarray((len(patient.RVstrain),5))
        measStrain[:,0:3]=patient.RVstrain
        measStrain[:,3]=patient.SVstrain
        measStrain[:,4]=patient.LVstrain


        # Find RV strain characteristics
        meanRV = measStrain.mean(1)

        idxMin = meanRV.argmin()
        idxConAt = -int(np.ceil(0.050/patient.RVtime[1]))
        idxCon50 = np.argwhere((range(len(meanRV))<idxMin) & (meanRV<.5*np.min(meanRV)))[0][0]
        idxRel50 = np.argwhere((range(len(meanRV))>idxMin) & (meanRV>.5*np.min(meanRV)))[0][0]

        # Get t0
        minErrorIdx = self.getIDX0(patient=patient,model_instance=model_instance,vectorData=vectorData)

        # Get strain for real cost functions
        if model_instance==[]:

            modStrain = np.array([vectorData['RvPatchRv1Ls'],vectorData['RvPatchRv2Ls'],vectorData['RvPatchRv3Ls'],vectorData['SvPatchSv1Ls'],vectorData['LvPatchLv1Ls']]).transpose()
            modStrain = np.concatenate( ( modStrain[minErrorIdx:,:], modStrain[1:minErrorIdx+1,:] ) )
            modStrain = modStrain / modStrain[minErrorIdx,:]
            modelTime = vectorData['Time']
        else:
            modStrain = model_instance.getComputedData('getCost2_modStrain'+str(minErrorIdx),model_instance.getStrain,['Rv1','Rv2','Rv3','Sv1','Lv1'],minErrorIdx)
            modelTime = model_instance.getTime()

        # Discretize Modelled Strain
        modStrainDiscretized = np.ndarray((len(patient.RVstrain),5))
        for iSeg in range(5):
            modStrainDiscretized[:,iSeg] = np.interp(patient.RVtime, modelTime, modStrain[:,iSeg])

        # Normalize LV strain
        GlobLVMeas = measStrain[:,3:5].mean(1).min()/100 + 1
        GlobLVMod  = modStrain[:,3:5].mean(1).min()/100 + 1
        measStrain[:,3:5] = (measStrain[:,3:5]/100 ) + 1
        measStrain[:,3:5] = (measStrain[:,3:5] - 1) / (GlobLVMeas-1) * (GlobLVMod-1)  + 1
        measStrain[:,3:5] = (measStrain[:,3:5]-1)*100

        # Calculate Error
        e2 = np.sum(( (modStrainDiscretized[:idxRel50,:]-measStrain[:idxRel50,:])/1)  **2) + ((modelTime[-1] - patient.RVtime[-1])/ 0.1)**2
        return e2

    def getNOut(self):
        return 1
