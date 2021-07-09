######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# Cost Function comparing RV3, IVS, and LV strain
######

from Cost import Cost
import numpy as np
import matplotlib.pyplot as plt

class CostStrain2(Cost):
    def __init__(self,costIDX0=[]):
        self.name = 'Strain2b'
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

        # Get Strain for comparison, resample model to measured temporal resolution and remove diastole part
        measStrain,modStrainDiscretized = self.getStrainForComparison(measStrain, modStrain,patient.RVtime,modelTime)

        # Calculate Error
        timeConst = 0.1
        strainConst = 1  / (patient.RVtime[len(modStrainDiscretized)]-patient.RVtime[0])

        e2 = np.mean(( (modStrainDiscretized-measStrain)/strainConst)  **2) + ((modelTime[-1] - patient.RVtime[-1])/ timeConst)**2

        useMLAPlim = True
        if useMLAPlim:
            mLAPlim = 25 #mmHg
            if not model_instance==[]:
                pressure = model_instance.getVector('La','Node','La','p')
                mLAP = np.mean(pressure) * 0.00750061683
                if mLAP > mLAPlim:
                    e2 = e2 + (mLAP-mLAPlim)**2


        return e2

    def getIDXRel50(self,measStrain):
        #Meas
        meanRV = measStrain.mean(1)
        idxMin = meanRV.argmin()
        idxRel50 = np.argwhere((range(len(meanRV))>idxMin) & (meanRV>.5*np.min(meanRV)))[0][0]
        return idxRel50


    def getStrainForComparison(self,measStrain,modStrain,measTime,modTime):
        # Discretize Modelled Strain
        modStrainDiscretized = np.ndarray((len(measStrain),5))
        for iSeg in range(5):
            modStrainDiscretized[:,iSeg] = np.interp(measTime, modTime, modStrain[:,iSeg])

        idxRel50 = self.getIDXRel50(measStrain)



        # Normalize LV strain
        GlobLVMeas = measStrain[:,3:5].mean(1).min()/100 + 1
        GlobLVMod  = modStrain[:,3:5].mean(1).min()/100 + 1

        if False: # Normalize meas
            measStrain[:,3:5] = (measStrain[:,3:5]/100 ) + 1
            measStrain[:,3:5] = (measStrain[:,3:5] - 1) / (GlobLVMeas-1) * (GlobLVMod-1)  + 1
            measStrain[:,3:5] = (measStrain[:,3:5] - 1) * 100
        else: #normalize Mod
            modStrainDiscretized[:,3:5] = (modStrainDiscretized[:,3:5] / 100 )
            modStrainDiscretized[:,3:5] = modStrainDiscretized[:,3:5] / (GlobLVMod-1) * (GlobLVMeas-1)
            modStrainDiscretized[:,3:5] = (modStrainDiscretized[:,3:5] ) * 100


        return measStrain[:idxRel50,:],modStrainDiscretized[:idxRel50,:]

    def getModelTime(self,model_instance,vectorData):
        # Init discretized modelled strain
        if model_instance==[]:
            return vectorData['Time']
        else:
            return model_instance.getTime()

    def getModelStrain(self,patient,model_instance,vectorData):
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
        return modStrain


    def plotValidation(self,patient=[],model_instance=[],vectorData=[]):
        modTime = self.getModelTime(model_instance,vectorData)
        modStrain = self.getModelStrain(patient,model_instance,vectorData)

        measTime   = patient.RVtime
        measStrain,measStrainMeanRV = self.measGetStrain(patient)

        plt.figure()
        plt.subplot(441)
        plt.plot(modTime,modStrain[:,:3])
        plt.title('Model Strain RV')
        plt.subplot(442)
        plt.plot(modTime,modStrain[:,3:])
        plt.title('Model Strain LV')

        plt.subplot(443)
        plt.plot(measTime,measStrain[:,:3])
        plt.title('Meas Strain RV')
        plt.subplot(444)
        plt.plot(measTime,measStrain[:,3:])
        plt.title('Meas Strain LV')

        ## STrain for comparison
        measStrain,modStrain = self.getStrainForComparison(measStrain,modStrain,measTime,modTime)

        plt.subplot(445)
        plt.plot(modStrain[:,:3])
        plt.title('Model Strain RV')
        plt.subplot(446)
        plt.plot(modStrain[:,3:])
        plt.title('Model Strain LV')

        plt.subplot(447)
        plt.plot(measStrain[:,:3])
        plt.title('Meas Strain RV')
        plt.subplot(448)
        plt.plot(measStrain[:,3:])
        plt.title('Meas Strain LV')


        plt.subplot(4,4,13)
        plt.plot(measStrain[:,:3] - modStrain[:,:3])
        plt.subplot(4,4,14)
        plt.plot(measStrain[:,3:] - modStrain[:,3:])


    def getNOut(self):
        return 1
