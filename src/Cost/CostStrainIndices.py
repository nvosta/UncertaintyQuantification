######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# Cost Function comparing RV3, IVS, and LV strain
######

from Cost import Cost
import numpy as np
import matplotlib.pyplot as plt
import _stiffness as stiffness
from scipy.signal import butter,filtfilt


class CostStrainIndices(Cost):
    def __init__(self,costIDX0=[]):
        self.indices = []
        self.weight = []
        self.uncertaintyStandardDeviation = []
        self.name = 'StrainIndices'
        self.costIDX0 = costIDX0
        self.doSimulation = False # false if cost is based on stoveVectorData

    def getStoreVectorData(self):
        return [['','','','Time'],['Rv','Patch','Rv1','Ls'],['Rv','Patch','Rv2','Ls'],['Rv','Patch','Rv3','Ls'],['Sv','Patch','Sv1','Ls'],['Lv','Patch','Lv1','Ls']]

    def getIDX0(self,patient,circadapt,vectorData=[]):
        if self.costIDX0==[]:
            timeMod = circadapt.getTime()
            iOnsetEnd = circadapt.onsetQRS() + 100

            error = [np.Infinity]*(iOnsetEnd)
            costMeas = self.getCostMeas(patient)

            stretchModel0 = self.getStretchModel(circadapt,patient=patient,idx0=0)
            strainModel0 = self.getStrainModel(circadapt,idx0=0)

            # Error is ALWAYS quadratic,
            optimumFound = False
            checkIdx = [0, circadapt.onsetQRS(), iOnsetEnd-1]
            while not(optimumFound):
                for i in checkIdx:
                    if error[i]==np.Infinity:
                        stretchModel = np.concatenate((stretchModel0[i:-1,:], stretchModel0[:(i+1),:]))
                        strainModel  = np.concatenate((strainModel0[i:-1,:], strainModel0[:(i+1),:]))
                        c = (self.getCostMod(circadapt,patient=patient,idx0=i,stretchModel=stretchModel,strainModel=strainModel,timeMod=timeMod)-costMeas)

                        # Correct for tCycle
                        tCycle = timeMod[-1]
                        for iC in range(len(self.indices)):
                            if self.indices[iC][0][:4]=='Time':
                                while c[iC] > 0.5*tCycle and not(np.isnan(c[iC])) and not(np.isinf(c[iC])):
                                    c[iC]=c[iC]-tCycle
                                while c[iC] < -0.5*tCycle and not(np.isnan(c[iC])) and not(np.isinf(c[iC])):
                                    c[iC]=c[iC]+tCycle

                        error[i] = sum(c**2)

                # next iteration
                # assume initial checkIDx has points left and right from optimum,
                #   - then if middle point is lowest error, check error around middle point
                if error[checkIdx[1]]<error[checkIdx[0]] and error[checkIdx[1]]<error[checkIdx[2]]:
                    checkIdx[0] = int((checkIdx[1] + checkIdx[0])/2)
                    checkIdx[2] = int((checkIdx[1] + checkIdx[2])/2)
                #   - if left point is lowest, check error around left
                elif error[checkIdx[0]]<error[checkIdx[2]]:
                    checkIdx[2] = checkIdx[1]
                    checkIdx[1] = int((checkIdx[0] + checkIdx[2])/2)
                #   - if right point is lowest, check error around right
                elif error[checkIdx[0]]<error[checkIdx[2]]:
                    checkIdx[0] = checkIdx[1]
                    checkIdx[1] = int((checkIdx[0] + checkIdx[2])/2)
                else:
                    checkIdx[0] = min(checkIdx[0]+1,checkIdx[1])
                    checkIdx[2] = max(checkIdx[2]-1,checkIdx[1])

                optimumFound = (checkIdx[0]==checkIdx[1]) and (checkIdx[1] == checkIdx[2])
            return checkIdx[0]

            #return np.argmin(error)
        else:
            return self.costIDX0.getIDX0(patient,circadapt,vectorData)

    def addIndices(self,index,loc,weight=1,uncertainty=0):
        '''
            Add multiple indices at same time
                index: string or list of strings
                            - PeakStrainPer
                            - StretchStrainPer
                            - ...
                loc: string or list of strings:
                            - RvApex,RvMid,RvBase,Lv,Sv
        '''
        if type(index)==list:
            for i in index:
                self.addIndices(i,loc,weight=weight,uncertainty=uncertainty)
        elif type(loc)==list:
            for l in loc:
                self.addIndices(index,l,weight=weight,uncertainty=uncertainty)
        else:
            self.indices.append([index,loc])
            self.weight.append(weight)
            self.uncertaintyStandardDeviation.append(uncertainty)
        return self

    def getStrainIndices(self,txt):
        strainIndex1 = -1
        strainIndex2 = -1
        if txt=='RvApex':
            strainIndex1 = 0
        elif txt=='RvMid':
            strainIndex1 = 1
        elif txt=='RvBase':
            strainIndex1 = 2
        elif txt=='Sv':
            strainIndex1 = 3
        elif txt=='Lv':
            strainIndex1 = 4
        elif txt=='GL':
            strainIndex1 = 5
        elif txt == 'GR' or txt == 'Rv':
            strainIndex1 = 6
        elif txt=='RvApexRvMid' or txt=='RvMidRvApex':
            strainIndex1 = 0
            strainIndex2 = 1
        elif txt=='RvApexRvBase' or txt=='RvBaseRvApex':
            strainIndex1 = 0
            strainIndex2 = 2
        elif txt=='RvBaseRvMid' or txt=='RvMidRvBase':
            strainIndex1 = 1
            strainIndex2 = 2
        elif txt=='RvApexGR' or txt=='GRRvApex':
            strainIndex1 = 0
            strainIndex2 = 6
        elif txt=='RvMidGR' or txt=='GRRvMid':
            strainIndex1 = 1
            strainIndex2 = 6
        elif txt=='RvBaseGR' or txt=='GRRvBase':
            strainIndex1 = 2
            strainIndex2 = 6
        elif txt=='RvApexLv' or txt=='LvRvApex':
            strainIndex1 = 0
            strainIndex2 = 4
        elif txt=='RvMidLv' or txt=='LvRvMid':
            strainIndex1 = 1
            strainIndex2 = 4
        elif txt=='RvBaseLv' or txt=='LvRvBase':
            strainIndex1 = 2
            strainIndex2 = 4
        elif txt=='RvApexSv' or txt=='SvRvApex':
            strainIndex1 = 0
            strainIndex2 = 3
        elif txt=='RvMidSv' or txt=='SvRvMid':
            strainIndex1 = 1
            strainIndex2 = 3
        elif txt=='RvBaseSv' or txt=='SvRvBase':
            strainIndex1 = 2
            strainIndex2 = 3
        elif txt=='LvSv' or txt=='SvLv':
            strainIndex1 = 3
            strainIndex2 = 4
        elif txt=='LvGR' or txt=='GRLv':
            strainIndex1 = 4
            strainIndex2 = 6
        elif txt=='SvGR' or txt=='GRSv' :
            strainIndex1 = 3
            strainIndex2 = 6
        elif txt=='GLGR' or txt=='GRGL':
            strainIndex1 = 5
            strainIndex2 = 6
        elif txt in ['', 'La', 'Ra', 'SyArt', 'PuArt']:
            pass
        else:
            strainIndex1 = np.nan
            strainIndex2 = np.nan

        return strainIndex1, strainIndex2

    def getIndexStrainContractionPercentage(self,stretch,num):
        argMax = np.argmax(stretch)
        argMin = np.argmin(stretch)

        maxStretch = np.max(stretch)
        minStretch = np.min(stretch)

        if num==0:
            return argMax
        if num>=100:
            return argMin

        stretch = np.concatenate( ( stretch[argMax:-1],stretch[:argMax] ) )

        orArgMax = argMax
        argMax = np.argmax(stretch)
        argMin = np.argmin(stretch)

        if argMax>argMin:
            argMax=0.0



        #if argMax < argMin: # Corrected, now stretch is translated suc h argmax=0

        # check if there is anything before the min
        idxBeforeMin = np.argwhere(
            (range(len(stretch))<argMin ) &
            (np.array(range(len(stretch)))>argMax ) &
            (stretch <= (maxStretch*(1-num/100.0) + minStretch * num/100.0))
            )



        if len(idxBeforeMin)==0 and False:
            plt.figure(99)
            plt.plot(stretch)
            idx1 = (range(len(stretch))<argMin )
            idx2 = (range(len(stretch))>argMax )
            idx3 = (stretch < (maxStretch*(1-num/100.0) + minStretch * num/100.0))
            X=np.array(range(len(stretch)))
            Y=X*0+1
            plt.plot(X[idx1],Y[idx1])
            plt.plot(X[idx2],Y[idx2]+0.01)
            plt.plot(X[idx3],Y[idx3]+0.02)
            print(stretch)
            print(maxStretch,minStretch,num,(maxStretch*(1-num/100.0) + minStretch * num/100.0))
            plt.show()
        if len(idxBeforeMin)==0:
            return np.nan

        idxBeforeMin = idxBeforeMin[0,0]+orArgMax
        if idxBeforeMin>len(stretch):
            idxBeforeMin=idxBeforeMin-len(stretch)
        return idxBeforeMin


        if False: # argMax > argMin:
            # check if there is anything before the min
            idxBeforeMin = np.argwhere(
                ( (range(len(stretch))>argMax )  ) &
                (stretch < (maxStretch*(1-num/100.0) + minStretch * num/100.0))
                )

            if len(idxBeforeMin)>0:
                return idxBeforeMin[0,0]
            else:
                idxAfterMin = np.argwhere(
                    (range(len(stretch))<argMin ) &
                    (stretch < (maxStretch*(1-num/100.0) + minStretch * num/100.0))
                    )

                if len(idxAfterMin)==0:
                    print(num)
                    plt.figure(99)
                    plt.plot(stretch)
                    idx1 = (range(len(stretch))<argMin )
                    idx2 = (stretch < (maxStretch*(1-num/100.0) + minStretch * num/100.0))
                    idx3 = (stretch < (maxStretch*(1-num/100.0) + minStretch * num/100.0))
                    X=np.array(range(len(stretch)))
                    Y=X*0+1
                    plt.plot(X[idx1],Y[idx1])
                    plt.plot(X[idx2],Y[idx2]+0.01)
                    plt.plot(X[idx3],Y[idx3]+0.02)
                    print(stretch)
                    print(maxStretch,minStretch,num,(maxStretch*(1-num/100.0) + minStretch * num/100.0))
                    #plt.show()

                # TODO: improve
                if len(idxAfterMin)==0:
                    return 0
                return idxAfterMin[-1,-1]


    def getCostStrain(self,time,strain,stretch,strainFromModel=False,patient=[]):
        cost = np.ndarray(len(self.indices))
        cost[:]=np.nan

        if np.any(np.isnan(stretch)) or np.any(np.isinf(stretch)):
            cost[:]=np.nan
            return cost


        for iIndex in range(len(self.indices)):
            strainIndex1, strainIndex2 = self.getStrainIndices(self.indices[iIndex][1])

            if np.isnan(strainIndex1) or np.isnan(strainIndex2):
                cost[:]=np.nan
                return cost

            maxStretch = np.max(stretch[:,:],0)
            stretchNormed = stretch[:,:]/maxStretch[None,:]

            # Calculate
            if self.indices[iIndex][0]=='PeakStrainPer':
                cost[iIndex]  = min(strain[:,strainIndex1])
            elif self.indices[iIndex][0]=='StretchStrainPer':
                cost[iIndex]  = max(strain[:,strainIndex1])
            elif self.indices[iIndex][0]=='NormMaxStretch':
                cost[iIndex]  = max(stretch[:,strainIndex1])/max(stretch[:,6])
            elif self.indices[iIndex][0]=='NormMinStretch':
                cost[iIndex]  = min(stretch[:,strainIndex1])/max(stretch[:,6])
            elif self.indices[iIndex][0]=='MaxStretchRefGRCon10':
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],10) # 10% contraction for GR
                cost[iIndex]  = max(stretch[:,strainIndex1])/stretch[idxRef,strainIndex1]
            elif self.indices[iIndex][0]=='MaxStretchRefGRCon50':
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],50) # 10% contraction for GR
                cost[iIndex]  = max(stretch[:,strainIndex1])/stretch[idxRef,strainIndex1]
            elif self.indices[iIndex][0]=='MaxStretchRefGRCon90':
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],90) # 10% contraction for GR
                cost[iIndex]  = max(stretch[:,strainIndex1])/stretch[idxRef,strainIndex1]
            elif self.indices[iIndex][0]=='MinStretchRefGRCon10':
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],10) # 10% contraction for GR
                cost[iIndex]  = min(stretch[:,strainIndex1])/stretch[idxRef,strainIndex1]
            elif self.indices[iIndex][0]=='MinStretchRefGRCon50':
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],50) # 10% contraction for GR
                cost[iIndex]  = min(stretch[:,strainIndex1])/stretch[idxRef,strainIndex1]
            elif self.indices[iIndex][0]=='MinStretchRefGRCon90':
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],90) # 10% contraction for GR
                cost[iIndex]  = min(stretch[:,strainIndex1])/stretch[idxRef,strainIndex1]




            elif self.indices[iIndex][0][0:7]=='Stretch' and self.indices[iIndex][0][7:11]!='Rate':
                timeFromNum = int(self.indices[iIndex][0][7:9])
                timeFromPhase = self.indices[iIndex][0][9:12]

                # Calc strain
                if timeFromPhase=='Con':
                    cost[iIndex]  = self.stretchOnTimeCon(time,stretch[:,strainIndex1],timeFromNum)
                if timeFromPhase=='Rel':
                    cost[iIndex]  = self.stretchOnTimeRel(time,stretch[:,strainIndex1],timeFromNum)


            elif self.indices[iIndex][0]=='TimeFromCon10GRToCon10':
                idx1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],10)
                idx2 = self.getIndexStrainContractionPercentage(stretch[:,6],10)
                cost[iIndex] = (time[idx1]-time[idx2])
            elif self.indices[iIndex][0]=='TimeFromCon10GRToCon25':
                idx1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],25)
                idx2 = self.getIndexStrainContractionPercentage(stretch[:,6],10)

                if False:
                    import matplotlib.pyplot as plt
                    plt.subplot(211)
                    plt.title(idx1)
                    plt.plot(stretch[:,strainIndex1])
                    plt.scatter(idx1,1)
                    plt.subplot(212)
                    plt.title(idx2)
                    plt.plot(stretch[:,6])
                    plt.scatter(idx2,1)
                    plt.show()
                cost[iIndex] = (time[idx1]-time[idx2])
            elif self.indices[iIndex][0]=='TimeFromCon10GRToCon50':
                idx1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],50)
                idx2 = self.getIndexStrainContractionPercentage(stretch[:,6],10)
                cost[iIndex] = (time[idx1]-time[idx2])
            elif self.indices[iIndex][0]=='TimeFromCon10GRToCon75':
                idx1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],75)
                idx2 = self.getIndexStrainContractionPercentage(stretch[:,6],10)
                cost[iIndex] = (time[idx1]-time[idx2])
            elif self.indices[iIndex][0]=='TimeFromCon10GRToCon90':
                #idx1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],90)
                #idx2 = self.getIndexStrainContractionPercentage(stretch[:,6],10)
                #cost[iIndex] = (time[idx1]-time[idx2])

                cost[iIndex] = (self.calcTimeToStretchCon(time,stretch[:,strainIndex1],90) -
                                self.calcTimeToStretchCon(time,stretch[:,6],10))

            elif self.indices[iIndex][0][0:4]=='Time':
                if strainIndex2==-1:

                    # Get Time difference between two points on one strain curve
                    n=4 # length of string, string always starts with Time

                    if self.indices[iIndex][0][n:n+5]== 'AtOns':
                        n=n+5
                        timeFromPhase = 'AtOns'
                    else:
                        timeFromNum = int(self.indices[iIndex][0][n:n+2])
                        n=n+2
                        if self.indices[iIndex][0][6:8] == 'GR':
                            strainIndexFrom = 6
                            n=n+2
                        else:
                            strainIndexFrom = strainIndex1

                        timeFromPhase = self.indices[iIndex][0][n:n+3]
                        n=n+3

                    if len(self.indices[iIndex][0])>n+2:
                        n=n+2
                        timeToNum = int(self.indices[iIndex][0][n:n+2])
                        n=n+2
                        timeToPhase = self.indices[iIndex][0][n:n+3]
                        n=n+3

                    # Calc strain
                    if timeFromPhase=='AtOns':
                        tFrom = self.findTimeOnsetAtrialKick(time,stretch[:,[3,4,6]],strainFromModel,patient=patient)
                        if np.isnan(tFrom) or np.isinf(tFrom):
                            cost[:]=np.nan
                            return cost

                        while tFrom>0 :
                            tFrom=tFrom-time[-1]
                    elif timeFromPhase=='Con':
                        tFrom = self.calcTimeToStretchCon(time,stretch[:,strainIndexFrom],timeFromNum)
                    elif timeFromPhase=='Rel':
                        tFrom = self.calcTimeToStretchRel(time,stretch[:,strainIndexFrom],timeFromNum)



                    if len(self.indices[iIndex][0])>11:
                        if timeToPhase=='Con':
                            tTo = self.calcTimeToStretchCon(time,stretch[:,strainIndex1],timeToNum)
                        if timeToPhase=='Rel':
                            tTo = self.calcTimeToStretchRel(time,stretch[:,strainIndex1],timeToNum)

                else:
                    timeFromNum = int(self.indices[iIndex][0][4:6])
                    timeFromPhase = self.indices[iIndex][0][6:9]
                    # find time of prestretch and min stretchs
                    sMod1  = stretchNormed[:,strainIndex1]
                    sMod2  = stretchNormed[:,strainIndex2]

                    # Calc strain
                    if timeFromPhase=='Con':
                        #tFrom = self.stretchOnTimeCon(time,sMod1,timeFromNum)
                        #tTo =   self.stretchOnTimeCon(time,sMod2,timeFromNum)

                        tFrom = self.getIndexStrainContractionPercentage(sMod1,timeFromNum)
                        tTo =   self.getIndexStrainContractionPercentage(sMod2,timeFromNum)

                        if np.isnan(tFrom) or np.isnan(tTo):
                            cost[:] = np.nan
                            return cost

                            print(sMod1,timeFromNum)
                            import matplotlib.pyplot as plt
                            plt.plot(sMod1)
                            plt.show()
                            print(tFrom)
                            print(strainIndex1,strainIndex2)

                        tFrom = time[tFrom]
                        tTo = time[tTo]

                    elif timeFromPhase=='Rel':
                        #tFrom = self.stretchOnTimeRel(time,sMod1,timeFromNum)
                        #tTo =   self.stretchOnTimeRel(time,sMod2,timeFromNum)
                        tFrom = self.getIndexStrainContractionPercentage(-sMod1,timeFromNum)
                        tTo =   self.getIndexStrainContractionPercentage(-sMod2,timeFromNum)

                        if np.isnan(tFrom) or np.isnan(tTo):
                            cost[:] = np.nan
                            return cost

                        tFrom = time[tFrom]
                        tTo = time[tTo]

                if len(self.indices[iIndex][0])<12 and strainIndex2 == -1:
                    cost[iIndex]  = tFrom
                else:
                    cost[iIndex]  = tTo - tFrom
                    if len(self.indices[iIndex][0])>11:
                        while cost[iIndex] < 0 and not(np.isnan(cost[iIndex])) and not(np.isinf(cost[iIndex])):
                            cost[iIndex]=cost[iIndex] + time[-1]
                        while cost[iIndex] > time[-1] and not(np.isnan(cost[iIndex])) and not(np.isinf(cost[iIndex])):
                            cost[iIndex]=cost[iIndex] - time[-1]
                    else:
                        while cost[iIndex] <-0.5*time[-1] and not(np.isnan(cost[iIndex])) and not(np.isinf(cost[iIndex])):
                            cost[iIndex]=cost[iIndex] + time[-1]
                        while cost[iIndex] > 0.5*time[-1] and not(np.isnan(cost[iIndex])) and not(np.isinf(cost[iIndex])):
                            cost[iIndex]=cost[iIndex] - time[-1]

            elif self.indices[iIndex][0]=='MinStretchRatioCon10':
                idxRef1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],10)
                idxRef2 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex2],10)
                cost[iIndex] = np.min(stretch[:,strainIndex1]/stretch[idxRef1,strainIndex1]) / np.min(stretch[:,strainIndex2]/stretch[idxRef2,strainIndex2])
            elif self.indices[iIndex][0]=='MinStretchRatioCon10GR':
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],10)
                cost[iIndex] = np.min(stretch[:,strainIndex1]/stretch[idxRef,strainIndex1]) / np.min(stretch[:,strainIndex2]/stretch[idxRef,strainIndex2])
            elif self.indices[iIndex][0]=='MaxStretchRatioCon10GR':
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],10)
                cost[iIndex] = np.max(stretch[:,strainIndex1]/stretch[idxRef,strainIndex1]) / np.max(stretch[:,strainIndex2]/stretch[idxRef,strainIndex2])

            elif self.indices[iIndex][0]=='TimeMax':
                cost[iIndex]  = time[np.argmax(stretch[:,strainIndex1])]
            elif self.indices[iIndex][0]=='TimeMin':
                cost[iIndex]  = time[np.argmin(stretch[:,strainIndex1])]



            elif ((len(self.indices[iIndex][0]) == len('StretchRateCon10GR')) or (len(self.indices[iIndex][0]) == len('StretchRateCon10'))   and
                self.indices[iIndex][0][:14]=='StretchRateCon'):
                if self.indices[iIndex][0][-2:] == 'GR':
                    cost[iIndex]  = self.getStretchRateAtCon(time,stretch[:,strainIndex1],int(self.indices[iIndex][0][14:16]),stretch[:,6])
                else:
                    cost[iIndex]  = self.getStretchRateAtCon(time,stretch[:,strainIndex1],int(self.indices[iIndex][0][14:16]))

            elif ((len(self.indices[iIndex][0]) == len('StretchRateNormedCon10GR')) or (len(self.indices[iIndex][0]) == len('StretchRateNormedCon10'))   and
                self.indices[iIndex][0][:20]=='StretchRateNormedCon'):
                if self.indices[iIndex][0][-2:] == 'GR':
                    cost[iIndex]  = self.getStretchRateNormedAtCon(time,stretch[:,strainIndex1],int(self.indices[iIndex][0][20:22]),stretch[:,6])
                else:
                    cost[iIndex]  = self.getStretchRateNormedAtCon(time,stretch[:,strainIndex1],int(self.indices[iIndex][0][20:22]))

            elif (self.indices[iIndex][0]=='StretchRatePeak'):
                cost[iIndex] = self.getPeakStretchRate(time,stretch[:,strainIndex1])
            elif (self.indices[iIndex][0]=='StretchRateNormedPeak'):
                s = stretch[:,strainIndex1]
                s = (s - np.min(s)) / (np.max(s) - np.min(s))
                cost[iIndex] = self.getPeakStretchRate(time,s)

            else: # handled elsewhere
                pass
                #cost[iIndex]  = 0
        return cost

    def getStrainMeas(self,patient):
        measStrain = np.ndarray((len(patient.RVstrain),7))
        measStrain[:,:3]=patient.RVstrain
        measStrain[:,3]=patient.SVstrain
        measStrain[:,4]=patient.LVstrain
        measStrain[:,5]=patient.GLstrain
        measStrain[:,6]=patient.GRstrain
        return measStrain

    def getStretchMeas(self,patient):
        measStrain = self.getStrainMeas(patient)
        measStretch = measStrain/100+1

        idxMax =np.argmax( measStretch[:,6])
        measStretch[:,:3] = measStretch[:,:3] / measStretch[idxMax,:3]
        measStretch[:,5]  = measStretch[:,5] / measStretch[idxMax,5]
        measStretch[:,6]  = measStretch[:,6] / measStretch[idxMax,6]
        return measStretch

    def getCostMeas(self,patient):
        # Calculate LV global stretch, assume segments have equal length
        measStrain = self.getStrainMeas(patient)
        measStretch = self.getStretchMeas(patient)

        timeMeas = patient.RVtime

        # Calculate Cost
        cost = self.getCostStrain(timeMeas,measStrain,measStretch,patient=patient)

        # Other indices
        for iIndex in range(len(self.indices)):
            # Calculate

            if self.indices[iIndex][0]=='EF' and (self.indices[iIndex][1]=='Lv' or self.indices[iIndex][1]=='LV'):
                if 'LVEF_echo' in patient.scalarData and patient.scalarData['LVEF_echo']>1:
                    cost[iIndex] = patient.scalarData['LVEF_echo']
                elif 'MRI_LVEF' in patient.scalarData:
                    cost[iIndex] = patient.scalarData['MRI_LVEF']
                else:
                    cost[iIndex] = np.nan;
            elif self.indices[iIndex][0]=='EDV' and (self.indices[iIndex][1]=='Lv' or self.indices[iIndex][1]=='LV'):
                if 'LVEDV_Echo' in patient.scalarData and patient.scalarData['LVEDV_Echo']>1:
                    cost[iIndex] = patient.scalarData['LVEDV_Echo']
                elif 'RVSV_MRI' in patient.scalarData:
                    cost[iIndex] = patient.scalarData['RVSV_MRI'] / patient.scalarData['MRI_LVEF'] * 100
                else:
                    cost[iIndex] = np.nan;


            elif self.indices[iIndex][0]=='tCycle':
                cost[iIndex] = patient.RVtime[-1]

            elif (len(self.indices[iIndex][0])=='mLAP50') and self.indices[iIndex][0][:4]=='mLAP':
                #mLAPlim = int(self.indices[iIndex][0][4:])
                cost[iIndex]=0
            elif (len(self.indices[iIndex][0])=='mLAPlin30') and self.indices[iIndex][0][:4]=='mLAP':
                #mLAPlim = int(self.indices[iIndex][0][4:])
                cost[iIndex]=0


        return cost

    def stretchOnTimeCon(self, time,stretch,num):
        t = self.calcTimeToStretchCon(time,stretch,num)
        return np.interp(t,time,stretch)

    def stretchOnTimeRel(self, time,stretch,num):
        t = self.calcTimeToStretchRel(time,stretch,num)
        while t>time[-1] and not(np.isnan(t)) and not(np.isinf(t)):
            t=t-time[-1]
        return np.interp(t,time,stretch)

    def butter_lowpass_filter(self,data, cutoff, fs, order):
        normal_cutoff = cutoff / (0.5*fs)
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def findTimeOnsetAtrialKick(self,t,s,strainFromModel=False,patient=[]):
        if np.shape(s)[0]==len(t):
            s=s.transpose()

        if strainFromModel:
            idxWindowOfInterest = range(np.argmax(s[0,:]))
        else:
            idxWindowOfInterest = int( 0.150/(t[1]-t[0]) )
            idxWindowOfInterest = list(range(len(t)-idxWindowOfInterest,len(t)))

        sWindowOfInterest = s[:,idxWindowOfInterest]

        tTot = 0
        for iS in range(np.size(s,0)):
            fs = 1 / (t[1]-t[0])
            sWindowOfInterest = self.butter_lowpass_filter(s[iS,:],5,fs,2)[idxWindowOfInterest]
            if len(sWindowOfInterest)<2:
                return np.nan
            #try:
            idx=np.argmax(np.diff(np.diff(sWindowOfInterest)))
            idx=idxWindowOfInterest[idx]
            tTot = tTot + t[idx]
            #except:
            #    return np.nan
        tTot = tTot / np.size(s,0)

        return tTot

    def calcTimeToStretchCon(self,time,stretch,num):
        idxMaxMeas = np.argmax(stretch)
        idxMinMeas = np.argmin(stretch)
        maxStretch = stretch[idxMaxMeas]
        minStretch = stretch[idxMinMeas]

        if idxMaxMeas > idxMinMeas:
            idxA = np.argwhere((stretch>maxStretch*(1-num/100.0) + minStretch * num/100.0) & ((range(len(stretch)) < idxMinMeas ) ) )
            idxB = np.argwhere((stretch>maxStretch*(1-num/100.0) + minStretch * num/100.0) & ((range(len(stretch)) > idxMaxMeas )) )

            if len(idxA)>0:
                idx1 = idxA[-1][0]
            else:
                if len(idxB)==0:
                    #print(stretch)
                    #plt.figure(99)
                    #plt.plot(stretch)
                    #plt.show()

                    # TODO: improve, now error
                    return np.nan
                idx1 = idxB[-1][0]
        else:
            idx1 = np.argwhere((stretch>maxStretch*(1-num/100.0) + minStretch * num/100.0) & (range(len(stretch)) < idxMinMeas ) )[-1][0]

        ## TODO FUTURE!!!
        if idx1==0:
            return time[idx1]
        elif idx1>=len(time)-1:
            idx1 = 0
        idx0 = idx1+1
        t = np.interp((maxStretch*(1-num/100.0) + minStretch * num/100.0), [stretch[idx0], stretch[idx1]], [time[idx0], time[idx1]])
        return t

    def calcTimeToStretchRel(self,time,stretch,num):
        maxStretch = np.max(stretch)
        minStretch = np.min(stretch)
        idxMaxMeas = np.argmin(stretch)
        idxMinMeas = np.argmin(stretch)

        idx1 = np.argwhere((stretch>maxStretch*num/100.0 + minStretch * (1-num/100.0)) & (range(len(stretch)) > idxMinMeas ))
        if len(idx1)>0:
            idx1=idx1[0][0]
        else:
            idx1 = np.argwhere((stretch>maxStretch*num/100.0 + minStretch * (1-num/100.0)) & (range(len(stretch)) < idxMaxMeas ))[0][0]
        ## TODO FUTURE!!!
        if idx1==0:
            return time[idx1]
        idx0 = idx1-1
        t = np.interp((maxStretch*num/100.0 + minStretch * (1-num/100.0)), [stretch[idx0], stretch[idx1]], [time[idx0], time[idx1]])

        # put all relaxation times after max strain
        if idx0<idxMaxMeas:
            t=t+time[-1]
        return t

    def getStrainModel(self,model_instance=[],vectorData=[],idx0=0):
        if model_instance==[]:
            strain = np.array([vectorData['RvPatchRv1Ls'],vectorData['RvPatchRv2Ls'],vectorData['RvPatchRv3Ls'],vectorData['RvPatchSv1Ls'],vectorData['RvPatchLv1Ls']])
            strain = np.concatenate( ( strain[idx0:,:], strain[1:idx0+1,:] ) )
            strain = strain / strain[idx0,:]
            return strain
        return model_instance.getStrain(['Rv1','Rv2','Rv3','Sv1','Lv1'],idx0)

    def getStretchModel(self,model_instance=[],vectorData=[],patient=[],idx0=0):
        strainModel = self.getStrainModel(model_instance)



        stretchModel = np.ndarray((len(strainModel),7))
        if model_instance==[]:
            stretchModel[:,:5] = np.array([vectorData['RvPatchRv1Ls'],vectorData['RvPatchRv2Ls'],vectorData['RvPatchRv3Ls'],vectorData['RvPatchSv1Ls'],vectorData['RvPatchLv1Ls']])
            stretchModel[:,:5] = stretchModel[:,:5] / stretchModel[0,:5]
        else:
            stretchModel[:,:5] = model_instance.getStretch(['Rv1','Rv2','Rv3','Sv1','Lv1'],idx0)

        # Calculate LV global stretch, assume segments have equal length
        if patient==[]:
            stretchModel[:,5] = (1 * stretchModel[:,3] + 3 * stretchModel[:,4] ) / 4
        else:
            stretchModel[:,5] = ( patient.nSV * stretchModel[:,3] + patient.nLV * stretchModel[:,4] ) / (patient.nLV + patient.nSV)
        stretchModel[:,6] = ( stretchModel[:,0] + stretchModel[:,1] + stretchModel[:,2] ) / 3

        idxMax =np.argmax( stretchModel[:,6])
        stretchModel[:,:3] = stretchModel[:,:3] / stretchModel[idxMax,:3]
        stretchModel[:,3]  = stretchModel[:,3] / stretchModel[idxMax,3]
        stretchModel[:,4]  = stretchModel[:,4] / stretchModel[idxMax,4]
        stretchModel[:,5]  = stretchModel[:,5] / stretchModel[idxMax,5]
        stretchModel[:,6]  = stretchModel[:,6] / stretchModel[idxMax,6]

        return stretchModel

    def getCostMod(self,model_instance=[],vectorData=[],patient=[],idx0=0,stretchModel=[],strainModel=[],timeMod=[]):
        cost = np.ndarray(len(self.indices))
        try:
            if strainModel==[]:
                strainModel = self.getStrainModel(model_instance=model_instance,vectorData=vectorData,idx0=idx0)
            if stretchModel==[]:
                stretchModel = self.getStretchModel(model_instance=model_instance,vectorData=vectorData,patient=patient,idx0=idx0)
            if timeMod == [] and model_instance==[]:
                timeMod = vectorData['Time']
            else:
                timeMod = model_instance.getTime()
        except:
            cost[:]=np.nan
            return cost

        # Calculate Cost
        try:
            cost = self.getCostStrain(timeMod,strainModel,stretchModel,strainFromModel=True)
        except:
            cost[:]=np.nan
            return cost

        # Other indices
        for iIndex in range(len(self.indices)):
            V = np.nan
            p = np.nan
            if self.indices[iIndex][0] in ['EF','EDV','ESV', 'max_volume', 'min_volume']:
                if self.indices[iIndex][1] in ['Lv', 'LV']:
                    V = np.array(model_instance.getVector('Lv','Cavity','Lv','V'))
                elif self.indices[iIndex][1] in ['Rv', 'RV']:
                    V = np.array(model_instance.getVector('Rv','Cavity','Rv','V'))
                elif self.indices[iIndex][1] in ['La', 'LA']:
                    V = np.array(model_instance.getVector('La','Cavity','La','V'))
                elif self.indices[iIndex][1] in ['Ra', 'RA']:
                    V = np.array(model_instance.getVector('Ra','Cavity','Ra','V'))
            if self.indices[iIndex][0] in ['max_pressure', 'min_pressure', 'mean_pressure']:
                if self.indices[iIndex][1] in ['Lv', 'LV']:
                    p = np.array(model_instance.getVector('Lv','Cavity','Lv','p'))
                elif self.indices[iIndex][1] in ['Rv', 'RV']:
                    p = np.array(model_instance.getVector('Rv','Cavity','Rv','p'))
                elif self.indices[iIndex][1] in ['La', 'LA']:
                    p = np.array(model_instance.getVector('La','Cavity','La','p'))
                elif self.indices[iIndex][1] in ['Ra', 'RA']:
                    p = np.array(model_instance.getVector('Ra','Cavity','Ra','p'))
                elif self.indices[iIndex][1] in ['SyArt']:
                    p = np.array(model_instance.getVector('CiSy','ArtVen','Art','p'))
                elif self.indices[iIndex][1] in ['PuArt']:
                    p = np.array(model_instance.getVector('CiPu','ArtVen','Art','p'))
            
            # Calculate
            if self.indices[iIndex][0]=='EF':
                cost[iIndex] = 100 - 100 * np.min(V)/np.max(V)
            elif self.indices[iIndex][0] in ['EDV', 'max_volume']:
                cost[iIndex] = np.max(V)*1e6
            elif self.indices[iIndex][0] in ['ESV', 'min_volume']:
                cost[iIndex] = np.min(V)*1e6
            elif self.indices[iIndex][0] in ['max_pressure']:
                cost[iIndex] = np.max(p)/133.322368
            elif self.indices[iIndex][0] in ['min_pressure']:
                cost[iIndex] = np.min(p)/133.322368
            elif self.indices[iIndex][0] in ['mean_pressure']:
                cost[iIndex] = np.mean(p)/133.322368
            elif self.indices[iIndex][0]=='tCycle':
                cost[iIndex] =  model_instance.getScalar('','','','tCycle')

            elif (len(self.indices[iIndex][0])=='mLAP50') and self.indices[iIndex][0][:4]=='mLAP':
                mLAPlim = int(self.indices[iIndex][0][4:])
                if model_instance==[]:
                    cost[iIndex]=0
                else:
                    pressure = model_instance.getVector('La','Node','La','p')
                    mLAP = np.mean(pressure)
                    cost[iIndex]=0 if mLAP < mLAPlim else np.Infinity
            elif (len(self.indices[iIndex][0])==len('mLAPlin25')) and self.indices[iIndex][0][:4]=='mLAP':
                mLAPlim = int(self.indices[iIndex][0][7:])
                if model_instance==[]:
                    cost[iIndex]=0
                else:
                    pressure = model_instance.getVector('La','Node','La','p')
                    mLAP = np.mean(pressure) * 0.00750061683
                    cost[iIndex]=0 if mLAP < mLAPlim else mLAP-mLAPlim
            elif self.indices[iIndex][0] in ['stiffness','contractility','compliance']:
                loc = np.nan
                # transform names to location in patch
                if self.indices[iIndex][1]=='Lv':
                    loc=[2]
                if self.indices[iIndex][1]=='Sv':
                    loc=[3]
                if self.indices[iIndex][1]=='RvApex':
                    loc=[4]
                if self.indices[iIndex][1]=='RvMid':
                    loc=[5]
                if self.indices[iIndex][1]=='RvBase':
                    loc=[6]
                
                
                cost[iIndex]=stiffness.get(model_instance, self.indices[iIndex][0], loc)[0]

        return cost

    def getCost(self,patient=[],model_instance=[],vectorData=[],modY=[],measY=[]):
        if measY==[]:
            measY = self.getCostMeas(patient)
        if modY==[] and model_instance==[]:
            modY = self.getCostMod(vectorData=vectorData)
        elif modY==[]:
            modY = self.getCostMod(model_instance=model_instance)
        c = (modY - measY)

        # Correct for tCycle
        if model_instance==[]:
            tCycle = vectorData['Time'][-1]
        else:
            tCycle = model_instance.getTime()[-1]
        for iC in range(len(self.indices)):
            if self.indices[iC][0][:4]=='Time':
                if c[iC] > 0.5*tCycle and not(np.isnan(c[iC])) and not(np.isinf(c[iC])):
                    c[iC]=c[iC]-tCycle
                if c[iC] < -0.5*tCycle and not(np.isnan(c[iC])) and not(np.isinf(c[iC])):
                    c[iC]=c[iC]+tCycle
        return c

    def getCost2(self,patient=[],model_instance=[],cost=[]):
        if cost==[]:
            cost=self.getCost(patient=patient,model_instance=model_instance)
        if np.any(np.isinf(cost)):
            return np.Infinity
        if np.any(np.isnan(cost)):
            return np.Infinity

        # Correct for tCycle
        if model_instance==[]:
            tCycle = vectorData['Time'][-1]
        else:
            tCycle = model_instance.getTime()[-1]
        for iC in range(len(self.indices)):
            if self.indices[iC][0][:4]=='Time':
                if cost[iC] > 0.5*tCycle and not(np.isnan(cost[iC])) and not(np.isinf(cost[iC])):
                    cost[iC]=cost[iC]-tCycle
                if cost[iC] < -0.5*tCycle and not(np.isnan(cost[iC])) and not(np.isinf(cost[iC])):
                    cost[iC]=cost[iC]+tCycle


        cost2 = ( cost / self.weight )**2
        return np.sum(cost2)


    def getNames(self,multiline=False,onlyIndexType=False,onlyIndexLocation=False):
        names = []
        for i in range(len(self.indices)):
            names.append(self.getName(i,multiline=multiline,onlyIndexType=onlyIndexType,onlyIndexLocation=onlyIndexLocation))
        return names

    def getName(self,i,multiline=False,onlyIndexType=False,onlyIndexLocation=False):
        p=self.indices[i]
        if onlyIndexLocation:
            return p[1]
        elif onlyIndexType:
            return p[0]
        elif multiline:
            return p[0]+'\n'+p[1]
        else:
            return p[0]+' '+p[1]

    def getNOut(self):
        return len(self.indices)

    def getCostSettings(self):
        return self.indices

    def showValidationIndices(self,patient=[],model_instance=[]):
        if not(patient==[]):
            costValues = self.getCostMeas(patient)
            # Collect data
            strain = self.getStrainMeas(patient)
            stretch = self.getStretchMeas(patient)
            time = patient.RVtime
        elif not(model_instance==[]):
            costValues = self.getCostMod(model_instance)
            # Collect data
            strain = self.getStrainModel(model_instance)
            stretch = self.getStretchModel(model_instance)
            time = model_instance.getTime()
        else:
            raise Exception('Error', 'No patient or model given')

        # Make Plots
        m=3
        n=5
        if len(self.indices)==1:
            m=1
            n=1
        elif len(self.indices)<7:
            m=2
            n=3
        iPlot = 0
        fig = plt.figure(iPlot+1)
        fig.patch.set_facecolor( (242/255,242/255,242/255) )
        plt.subplots_adjust(hspace=1)

        for iIndex in range(len(self.indices)):
            if iIndex-iPlot*m*n + 1> m*n:
                iPlot = iPlot + 1
                fig = plt.figure(iPlot+1)
                fig.patch.set_facecolor( (242/255,242/255,242/255) )
            plt.subplot(m,n,iIndex-iPlot*m*n+1)
            plt.subplots_adjust(hspace=0.75)


            # Plot Index
            strainIndex1, strainIndex2 = self.getStrainIndices(self.indices[iIndex][1])
            if self.indices[iIndex][0] == 'NormMaxStretch':
                X = time[np.argmax(stretch[:,strainIndex1])]
                Y = costValues[iIndex]
                plt.plot(time,stretch[:,strainIndex1])
                plt.scatter(X,Y)
            elif self.indices[iIndex][0] == 'NormMinStretch':
                X = time[np.argmin(stretch[:,strainIndex1])]
                Y = costValues[iIndex]
                plt.plot(time,stretch[:,strainIndex1])
                plt.scatter(X,Y)
            elif len(self.indices[iIndex][0]) == len('TimeFromCon10GRToCon10') and self.indices[iIndex][0][:-2] == 'TimeFromCon10GRToCon':
                num = int(self.indices[iIndex][0][-2:])
                idx1 = self.getIndexStrainContractionPercentage(stretch[:,6],10)
                idx2 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],num)

                X = [time[idx1], time[idx1] + costValues[iIndex]]
                Y = [np.min(stretch),np.max(stretch)]
                plt.plot(time,stretch[:,strainIndex1])
                plt.plot([X[0],X[0]],Y,color='k')
                plt.plot([X[1],X[1]],Y,color='k')

            elif len(self.indices[iIndex][0]) == len('Time10ConTo25Con') and self.indices[iIndex][0][:4] == 'Time' and self.indices[iIndex][0][9:11] == 'To':
                if self.indices[iIndex][0][4:9] =='AtOns':
                    idx1 = self.findTimeOnsetAtrialKick(time,stretch[:,[3,4,6]],strainFromModel=patient==[],patient=patient)
                else:
                    num1 = int(self.indices[iIndex][0][4:6])
                    if self.indices[iIndex][0][6:9] == 'Con':
                        idx1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],num1)
                        idx1 = time[idx1]
                    elif self.indices[iIndex][0][6:9] == 'Rel':
                        idx1 = self.getIndexStrainContractionPercentage(-np.array(stretch[:,strainIndex1]),num1)
                        idx1 = time[idx1]

                X = [idx1,idx1 + costValues[iIndex]]
                Y = [np.min(stretch),np.max(stretch)]

                t = np.concatenate((np.array(time[:-1])-time[-1], time))
                s = np.concatenate((stretch[:,strainIndex1], stretch[1:,strainIndex1]))

                plt.plot(t,s)
                plt.plot([X[0],X[0]],Y,color='k')
                plt.plot([X[1],X[1]],Y,color='k')

            elif len(self.indices[iIndex][0]) == len('Time10Con') and self.indices[iIndex][0][:4]=='Time':
                num1 = int(self.indices[iIndex][0][4:6])
                if self.indices[iIndex][0][6:9] == 'Con':
                    #idx1 = self.stretchOnTimeCon(time,stretch[:,strainIndex1],num1)
                    idx1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],num1)
                    idx1 = time[idx1]
                elif self.indices[iIndex][0][6:9] == 'Rel':
                    idx1 = self.stretchOnTimeRel(time,stretch[:,strainIndex1],num1)
                    idx1 = self.getIndexStrainContractionPercentage(-np.array(stretch[:,strainIndex1]),num1)
                    idx1 = time[idx1]
                X = [idx1,idx1 + costValues[iIndex]]
                Y = [np.min(stretch),np.max(stretch)]
                plt.plot(time,stretch[:,strainIndex1])
                plt.plot(time,stretch[:,strainIndex2])
                plt.plot([X[0],X[0]],Y,color='k')
                plt.plot([X[1],X[1]],Y,color='k')

            elif len(self.indices[iIndex][0]) == len('Time10GRConTo25Con') and self.indices[iIndex][0][:4] == 'Time' and self.indices[iIndex][0][11:13] == 'To':
                num1 = int(self.indices[iIndex][0][4:6])
                if self.indices[iIndex][0][8:11] == 'Con':
                    idx1 = self.getIndexStrainContractionPercentage(stretch[:,6],num1)
                    idx1 = time[idx1]
                elif self.indices[iIndex][0][8:11] == 'Rel':
                    idx1 = self.getIndexStrainContractionPercentage(-np.array(stretch[:,6]),num1)
                    idx1 = time[idx1]

                X = [idx1,idx1 + costValues[iIndex]]
                Y = [np.min(stretch),np.max(stretch)]
                plt.plot(time,stretch[:,strainIndex1])
                plt.plot([X[0],X[0]],Y,color='k')
                plt.plot([X[1],X[1]],Y,color='k')



            elif ((len(self.indices[iIndex][0]) == len('MaxStretchRefGRCon10')) and self.indices[iIndex][0][:13]=='MaxStretchRef'):
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],int(self.indices[iIndex][0][-2:])) # 10% contraction for GR
                #cost[iIndex]  = max(stretch[:,strainIndex1])/stretch[idxRef,strainIndex1]

                plt.plot(time,stretch[:,strainIndex1]/stretch[idxRef,strainIndex1])
                plt.scatter(time[np.argmax(stretch[:,strainIndex1]/stretch[idxRef,strainIndex1])],costValues[iIndex])
                plt.scatter(time[idxRef],costValues[iIndex])
            elif ((len(self.indices[iIndex][0]) == len('MinStretchRefGRCon10')) and self.indices[iIndex][0][:13]=='MinStretchRef'):
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],int(self.indices[iIndex][0][-2:])) # 10% contraction for GR
                #cost[iIndex]  = max(stretch[:,strainIndex1])/stretch[idxRef,strainIndex1]

                plt.plot(time,stretch[:,strainIndex1]/stretch[idxRef,strainIndex1])
                plt.scatter(time[np.argmin(stretch[:,strainIndex1]/stretch[idxRef,strainIndex1])],costValues[iIndex])
                plt.scatter(time[idxRef],costValues[iIndex])

            elif (self.indices[iIndex][0]=='MinStretchRatioCon10'):
                idxRef1 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex1],int(self.indices[iIndex][0][-2:]))
                idxRef2 = self.getIndexStrainContractionPercentage(stretch[:,strainIndex2],int(self.indices[iIndex][0][-2:]))
                plt.plot(time,stretch[:,strainIndex1]/stretch[idxRef1,strainIndex1])
                plt.plot(time,stretch[:,strainIndex2]/stretch[idxRef2,strainIndex2])
                plt.scatter(time[np.argmin(stretch[:,strainIndex1])],np.min(stretch[:,strainIndex1]/stretch[idxRef1,strainIndex1]))
                plt.scatter(time[np.argmin(stretch[:,strainIndex2])],np.min(stretch[:,strainIndex1]/stretch[idxRef1,strainIndex1])/costValues[iIndex])
            elif (self.indices[iIndex][0]=='MinStretchRatioCon10GR'):
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],int(self.indices[iIndex][0][-4:-2]))
                plt.plot(time,stretch[:,strainIndex1]/stretch[idxRef,strainIndex1])
                plt.plot(time,stretch[:,strainIndex2]/stretch[idxRef,strainIndex2])
                plt.scatter(time[np.argmin(stretch[:,strainIndex1])],np.min(stretch[:,strainIndex1]/stretch[idxRef,strainIndex1]))
                plt.scatter(time[np.argmin(stretch[:,strainIndex2])],np.min(stretch[:,strainIndex1]/stretch[idxRef,strainIndex1])/costValues[iIndex])

            elif (self.indices[iIndex][0]=='MaxStretchRatioCon10GR'):
                idxRef = self.getIndexStrainContractionPercentage(stretch[:,6],int(self.indices[iIndex][0][-4:-2]))
                plt.plot(time,stretch[:,strainIndex1]/stretch[idxRef,strainIndex1])
                plt.plot(time,stretch[:,strainIndex2]/stretch[idxRef,strainIndex2])
                plt.scatter(time[np.argmax(stretch[:,strainIndex1])],np.max(stretch[:,strainIndex1]/stretch[idxRef,strainIndex1]))
                plt.scatter(time[np.argmax(stretch[:,strainIndex2])],np.max(stretch[:,strainIndex1]/stretch[idxRef,strainIndex1])/costValues[iIndex])


            elif ((len(self.indices[iIndex][0]) == len('StretchRateCon10GR')) or (len(self.indices[iIndex][0]) == len('StretchRateCon10'))   and
                self.indices[iIndex][0][:14]=='StretchRateCon'):
                stretchrate = np.ndarray(101)
                iLoc = 6 if self.indices[iIndex][0][-2:] == 'GR' else strainIndex1

                for iS in range(101):
                    stretchrate[iS]=self.getStretchRateAtCon(time,stretch[:,strainIndex1],iS,stretch[:,iLoc])

                plt.plot(range(101),stretchrate)
                plt.scatter(int(self.indices[iIndex][0][14:16]),costValues[iIndex])

            elif ((len(self.indices[iIndex][0]) == len('StretchRateNormedCon10GR')) or (len(self.indices[iIndex][0]) == len('StretchRateNormedCon10'))   and
                self.indices[iIndex][0][:20]=='StretchRateNormedCon'):
                stretchrate = np.ndarray(101)
                iLoc = 6 if self.indices[iIndex][0][-2:] == 'GR' else strainIndex1

                for iS in range(101):
                    stretchrate[iS]=self.getStretchRateNormedAtCon(time,stretch[:,strainIndex1],iS,stretch[:,iLoc])

                plt.plot(range(101),stretchrate)
                plt.scatter(int(self.indices[iIndex][0][20:22]),costValues[iIndex])

            elif (self.indices[iIndex][0]=='StretchRatePeak'):
                stretchrate = np.ndarray(101)
                iLoc = strainIndex1

                for iS in range(101):
                    stretchrate[iS]=self.getStretchRateAtCon(time,stretch[:,strainIndex1],iS,stretch[:,iLoc])

                plt.plot(range(101),stretchrate)
                plt.scatter(np.argmin(stretchrate),costValues[iIndex])

            elif (self.indices[iIndex][0]=='StretchRateNormedPeak'):
                stretchrate = np.ndarray(101)
                iLoc = strainIndex1

                for iS in range(101):
                    stretchrate[iS]=self.getStretchRateNormedAtCon(time,stretch[:,strainIndex1],iS,stretch[:,iLoc])

                plt.plot(range(101),stretchrate)
                plt.scatter(np.argmin(stretchrate),costValues[iIndex])

            tit = self.indices[iIndex][0] + '\n' + self.indices[iIndex][1] + '\n' + '{:04.3f}'.format(costValues[iIndex])
            plt.title(tit)

        plt.show()

    def getStretchRateAtCon(self,time,stretch,num,stretchNum=[]):
        if stretchNum==[]:
            idxInterpolated = self.getInterpContractionPercentages(stretch,contractionPercentages=num)
        else:
            idxInterpolated = self.getInterpContractionPercentages(stretchNum,contractionPercentages=num)

        windowWidth = 9

        timeM = np.array(time[:windowWidth]) - np.mean(time[:windowWidth])

        # calculate derivative on idxGR
        window = np.array(range(windowWidth)) - np.floor(windowWidth/2)
        windowIDX = window + idxInterpolated
        windowIDX[windowIDX<0] = windowIDX[windowIDX<0] + len(stretch)
        stretchM = np.interp(windowIDX,range(len(stretch)),stretch)

        phi = np.ones((windowWidth,2))
        phi[:,1] = timeM

        c = np.linalg.inv(phi.transpose().dot(phi)).dot(phi.transpose()).dot(stretchM)

        return c[1]

    def getPeakStretchRate(self,time,stretch):

        #start
        maxRate = 0
        windowWidth = 9
        timeM = np.array(time[:windowWidth]) - np.mean(time[:windowWidth])



        allIndex = np.argwhere(
            (range(len(stretch)) > np.argmax(stretch) ) & (range(len(stretch)) < np.argmin(stretch) )
        )

        if len(allIndex)==0:
            allIndex = np.argwhere(
                (range(len(stretch)) > np.argmax(stretch) ) | (range(len(stretch)) < np.argmin(stretch) )
            )




        phi = np.ones((windowWidth,2))
        phi[:,1] = timeM



        for iIndex in allIndex:
            window = np.array(range(windowWidth)) - np.floor(windowWidth/2)
            windowIDX = window + iIndex
            windowIDX[windowIDX<0] = windowIDX[windowIDX<0] + len(stretch)

            stretchM = np.interp(windowIDX,range(len(stretch)),stretch)
            c = np.linalg.inv(phi.transpose().dot(phi)).dot(phi.transpose()).dot(stretchM)


            maxRate = c[1] if c[1] < maxRate else maxRate
        return maxRate

    def getStretchRateNormedAtCon(self,time,stretch,num,stretchNum=[]):
        stretch = (stretch - np.min(stretch)) / (np.max(stretch)-np.min(stretch))
        return self.getStretchRateAtCon(time,stretch,num,stretchNum)


    def getInterpContractionPercentages(self,vectorData,contractionPercentages = np.array(range(101))):
        ''' Interpolate time IDX to contraction, assuming argMax < argMin '''
        # find min/max
        argMax = np.argmax(vectorData)
        argMin = np.argmin(vectorData)
        Max = np.max(vectorData)
        Min = np.min(vectorData)

        if argMax>argMin:
            vd = np.concatenate( (vectorData[argMax:], vectorData[:argMin] ))
            argMax = argMax - len(vectorData)
        else:
            vd = vectorData[argMax:argMin]

        # calculate contraction
        contraction = 100 - 100*(vd - Min) / (Max - Min)
        timeIDX = np.array(range(len(contraction))) + argMax

        # interpolate
        return np.interp(contractionPercentages,contraction,timeIDX)
