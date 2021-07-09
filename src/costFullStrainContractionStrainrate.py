from Cost import Cost
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks

class costFullStrainContractionStrainrate(Cost):
    def __init__(self,costIDX0=[]):
        self.name = 'costFullStrainContraction'
        self.doSimulation = True # false if cost is based on stoveVectorData

        self.rateconst = [0, 0, 0, 0, 0]
        self.rateconstFull = [0, 0, 0, 0, 0]
        self.segment_difference_const = 0
        self.segment_difference_full_range_const = 0

        self.useLVEF = False
        self.constLVEF = 2.5

        self.useEDV = False
        self.constEDV = 25
        
        self.useRVD = False
        self.constRVD = 5 # mm

        self.useMaxMLAP = False
        self.limMaxMLAP = 15*133.322368
        self.constMaxMLAP = 1*133.322368
        
        self.useMaxMRAP = False
        self.limMaxMRAP = 10*133.322368
        self.constMaxMRAP = 1*133.322368

        self.useNonNegativeVolumes = False

        self.useMaxTimeToPeakstrain = False
        self.maxTimeToPeakstrain = 0.1
        self.constMaxTimeToPeakstrain = 0.01

        self.useLVdiastolicPositiveStrain = False

        self.normalize_modelled_strain = False

        self.strainFac = [1,1,1,3,3] # RV1,2,3, LV, SV
        self.strainFacFull = [0,0,0,0,0]

        self.facGRstrain = 0

        self.splitRVLV = True

        self.useMink1 = False
        self.mink1 = 1.01
        self.constMink1 = 0.0001

        self.useMinSfAct = False
        self.minSfACt = 0.001
        self.constMinSfACt = 0.001
        
        self.useMaxSfAct = False
        self.maxSfAct = 500000
        self.constMaxSfAct = 1
        
        self.useMaxQ0 = False
        self.maxQ0 = 8.5e-05 * 2.5
        self.constQ0 = 8.5e-05/5
        
        self.useMinAVdelay = False
        self.minAVfac = 0.25 # relative to model_instance.getScalar('','','','TauAv')
        self.constMinAVdelay = 0.01

        self.useMaxDt = False
        self.maxDtFac = 0.5
        self.constMaxDt = 0.1
        
        self.useMaxRVEF = False
        self.maxRVEF = 75 # relative to model_instance.getScalar('','','','TauAv')
        self.constMaxRVEF = 1
        
        self.useNoBulging = False
        self.minCm = 0 # relative to model_instance.getScalar('','','','TauAv')
        self.constMinCm = 0.01
        
        self.useGLSRV = False
        self.constGLSRV = 2.5
        
        self.useAmRefRatio = False
        self.maxAmRefRatio = 1/2 # SV / RV
        self.constAmRefRatio = 0.01
        
        self.useMaxRate = False
        self.maxRate = 1


        self.useStrainTime = False
        self.constStrainTime = 0.025
        self.facStrainTime = [1,1,1,3,3] # RV1,2,3, LV, SV


    def getIDX0(self,patient,circadapt):

        strainMeas = self.getStrainMeas(patient)
        strainMod  = self.getStrainModel(model_instance=circadapt,idx0='onsetQRS')
        timeMeas    = patient.RVtime
        timeMod    = circadapt.getTime()

        # RV
        tAtrialMeas = self.findTimeOnsetAtrialKick(timeMeas, strainMeas[:,:3])
        tAtrialMod = self.findTimeOnsetAtrialKick(timeMod, strainMod[:,:3])
        dIDX = int((tAtrialMod - tAtrialMeas)/0.002)

        return circadapt.onsetQRS() + dIDX

    def getNOut(self):
        return 1

    def butter_lowpass_filter(self,data, cutoff, fs, order):
        normal_cutoff = cutoff / (0.5*fs)
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def findTimeOnsetAtrialKick(self,t,s,strainFromModel=False,patient=[], maxTime=-1):
        if maxTime < 0:
            maxTime = t[-1]*0.200
            
        if np.shape(s)[0]==len(t):
            s=s.transpose()

        idxWindowOfInterest = int( maxTime/(t[1]-t[0]) )
        idxWindowOfInterest = list(range(len(t)-idxWindowOfInterest,len(t)))

        sWindowOfInterest = s[:,idxWindowOfInterest]
        
        segment_weights = np.ones(np.size(s,0))
        if np.size(s,0)==3: # correct for bad RV segments
            strain_n_above_zero = np.sum(sWindowOfInterest>0, axis=1)
            strain_fac_above_zero = strain_n_above_zero / sWindowOfInterest.shape[1]
            segment_weights = 1 - strain_fac_above_zero

        tTot = 0
        for iS in range(np.size(s,0)):
            fs = 1 / (t[1]-t[0])
            sWindowOfInterest = self.butter_lowpass_filter(s[iS,:],5,fs,2)[idxWindowOfInterest]
            if len(sWindowOfInterest)<2:
                return np.nan
            try:
                #idx=np.argmax(np.diff(np.diff(sWindowOfInterest)))
                
                #always find latest peak
                peaks = find_peaks(np.diff(np.diff(sWindowOfInterest)))
                if len(peaks[0])>0:
                    idx = np.max(peaks[0])
                    idx=idxWindowOfInterest[idx]
                    tTot = tTot + t[idx]*segment_weights[iS]
                else:
                    segment_weights[iS] = 0 # get information from other segments
            except:
                return np.nan
        if np.sum(segment_weights)==0:
            return t[idxWindowOfInterest[0]]
            #raise Exception('Can not find atrial kick in strain segment' )
        
        tTot = tTot / np.sum(segment_weights)
        
        return tTot

    def getStrainMeas(self,patient):
        measStrain = np.ndarray((len(patient.RVstrain),7))
        measStrain[:,:3]=patient.RVstrain
        measStrain[:,3]=patient.SVstrain
        measStrain[:,4]=patient.LVstrain
        measStrain[:,5]=patient.GRstrain
        measStrain[:,6]=patient.GLstrain
        return measStrain

    def getStrainModel(self,model_instance=[],vectorData=[],idx0=0):
        if model_instance==[]:
            strain = np.array([vectorData['RvPatchRv1Ls'],vectorData['RvPatchRv2Ls'],vectorData['RvPatchRv3Ls'],vectorData['RvPatchSv1Ls'],vectorData['RvPatchLv1Ls']])
            strain = np.concatenate( ( strain[idx0:,:], strain[1:idx0+1,:] ) )
            strain = strain / strain[idx0,:]
            return strain
        try:
            return model_instance.getStrain(['Rv1','Rv2','Rv3','Sv1','Lv1'],idx0)
        except:
            return np.nan

    def getCost(self,patient=[],model_instance=[],modY=[],measY=[],vectorData=[]):
        e = self.getCost2(patient=patient,model_instance=model_instance,vectorData=vectorData)
        print('cost2',e)
        e = np.sqrt(e)
        print('cost',e)
        return e



    def getCost2(self,patient=[],model_instance=[],vectorData=[]):
        e = 0

        strainMeas = self.getStrainMeas(patient)
        timeMeas   = patient.RVtime
        try:
            strainMod, tDiff  = self.getModelStrainIDX0corrected(model_instance=model_instance,strainMeas=strainMeas,timeMeas=timeMeas)
        except:
            return np.nan
        
        timeMod    = model_instance.getTime()
        
        idxMeasDiff = np.max([0, np.argmax(np.array(timeMeas)>np.diff(tDiff))  ])

        tDiff = np.abs(np.diff(tDiff))
        tDiff = np.min([tDiff, np.abs(timeMeas[-1] - tDiff)])

        if tDiff > 0.01:
            e = e + ((tDiff-0.025) / 0.01)**2

        if np.any(np.isnan(strainMod)):
            return np.nan

        strainMod = self.getDiscretizedStrain(timeMod,strainMod,timeMeas)
        timeMod=timeMeas

        X = 0
        for iSeg in range(5):
            X = np.max([X,  np.argmin(strainMeas[:,iSeg]) + 1])
        X=X+int(np.ceil(0.05/(timeMod[1]-timeMod[0])))

        for iSeg in range(5):
            # X = np.argmin(strainMeas[:,iSeg]) + int(0.1/(timeMod[1]-timeMod[0]))
            try:
                e = e + self.getCost2SingleStrain(timeMod[:X],strainMod[:X,iSeg],strainMeas[:X,iSeg],
                                                  const=self.strainFac[iSeg],rateconst=self.rateconst[iSeg])
                e = e + self.getCost2SingleStrain(timeMod[:],strainMod[:,iSeg],strainMeas[:,iSeg],
                                                  const=self.strainFacFull[iSeg], rateconst=self.rateconstFull[iSeg])
            except:
                return np.nan
        
        meanStrain = np.mean(strainMeas[:,:],axis=1)
        #X = np.array(range(len(timeMod)))
        #X = np.argwhere(  (X>np.argmin(meanStrain)) & (meanStrain > 0.5*np.min(meanStrain))  )[0,0]

        if self.facGRstrain>0:
            strainGR = np.mean(strainMod[:X,:3],axis=1)
            e = e + self.facGRstrain * self.getCost2SingleStrain(timeMod[:X],strainGR,strainMeas[:X,5],const=self.strainFac[iSeg], rateconst=0)



        if self.segment_difference_const>0:
            diff_index = [[0,1],[1,2],[0,2],[3,4]]
            for d in diff_index:
                e = e + self.segment_difference_const * self.getCost2SingleStrainDiff(timeMod[:X],strainMod[:X,d], strainMeas[:X,d])
            if False:
                # normalize for meanLV - meanRV
                if X+idxMeasDiff > strainMod.shape[0]:
                    e = e + 100 # simulation not healthy
                else:
                    meanMod = np.concatenate((np.mean(strainMod[idxMeasDiff:X+idxMeasDiff,0:3], axis=1).reshape(-1,1), 
                                              np.mean(strainMod[:X,3:5], axis=1).reshape(-1,1)), axis=1)
                    meanMeas = np.concatenate((np.mean(strainMeas[:X,0:3], axis=1).reshape(-1,1), 
                                               np.mean(strainMeas[:X,3:5], axis=1).reshape(-1,1)), axis=1)
                    meanMod[:,0] = meanMod[:,0] / np.min(meanMod[:,0]) * np.min(meanMod[:,1])
                    meanMeas[:,0] = meanMeas[:,0] / np.min(meanMeas[:,0]) * np.min(meanMeas[:,1])
                    e = e + 0.5 * self.segment_difference_const * self.getCost2SingleStrainDiff(timeMod[:X],np.mean(strainMod[:X,0:3], axis=1), np.mean(strainMeas[:X,3:5], axis=1))

        if self.segment_difference_full_range_const>0:
            diff_index = [[0,1],[1,2],[0,2],[3,4]]
            for d in diff_index:
                e = e + self.segment_difference_full_range_const * self.getCost2SingleStrainDiff(timeMod,strainMod[:,d], strainMeas[:,d])



        if self.useMaxTimeToPeakstrain:
            for iSeg in range(5):
                argmax_mod = np.argmin(strainMod[:,iSeg])
                argmax_meas = np.argmin(strainMeas[:,iSeg])
                time_diff = np.abs(timeMod[argmax_mod]-timeMod[argmax_meas])
                time_diff = np.max([time_diff-self.maxTimeToPeakstrain, 0])

                e = e + (time_diff/self.constMaxTimeToPeakstrain)**2


        if self.useStrainTime:
            iSeg = 2
            #self.getCost2StrainTime(timeMod[:],strainMod[:,iSeg],strainMeas[:,iSeg])
            for iSeg in range(5):
                e = e + self.facStrainTime[iSeg] * self.getCost2StrainTime(timeMod[:],strainMod[:,iSeg],strainMeas[:,iSeg])


        if False:
            meanStrain = np.mean(strainMod[:,:],axis=1)
            X1 = np.array(range(len(timeMod)))
            X1 = np.argwhere(  (X1>np.argmin(meanStrain)) & (meanStrain > 0.5*np.min(meanStrain))  )
            if len(X1)>0:
                X1=X1[0,0]
            else:
                X1=len(timeMod)-1
    
            e = e + (np.max([0, np.abs(timeMod[X1]-timeMod[X])-0.05])/0.1)**2

        if self.useLVEF:
            e = e + self.getErrorLVEF(patient,model_instance)
        if self.useEDV:
            e = e + self.getErrorEDV(patient,model_instance)
        if self.useRVD:
            e = e + self.getErrorRVD(patient,model_instance)
        if self.useMaxMLAP:
            e = e + self.getErrorMLAP(patient,model_instance)
        if self.useMaxMRAP:
            e = e + self.getErrorMRAP(patient,model_instance)
        if self.useNonNegativeVolumes:
            e = e + self.getErrorNonNegativeVolumes(model_instance)
        if self.useLVdiastolicPositiveStrain:
            e = e + self.getErrorLVdiastolicPositiveStrain(strainMod)
        if self.useMink1:
            e = e + self.getErrorK1(model_instance)
        if self.useMinSfAct:
            e = e + self.getErrorSfAct(model_instance)
        if self.useMaxSfAct:
            e = e + self.getErrorMaxSfAct(model_instance)
        if self.useMaxQ0:
            e = e + self.getErrorMaxQ0(model_instance)
        if self.useMinAVdelay:
            e = e + self.getErrorMinAVdelay(model_instance)
        if self.useMaxDt:
            e = e + self.getErrorMaxDT(model_instance)
        if self.useMaxRVEF:
            e = e + self.getErrorMaxRVEF(model_instance)
        if self.useNoBulging:
            e = e + self.getErrorSeptalBulging(model_instance)
        if self.useGLSRV:
            meanMod = np.mean(strainMod[:,0:3], axis=1)
            meanMeas = np.mean(strainMeas[:,0:3], axis=1)
            e = e + self.getErrorGLSRV(meanMod, meanMeas)
        if self.useAmRefRatio:
            e = e + self.getErrorAmRefRatio(model_instance)
            
        return e
    
    def getErrorGLSRV(self, RVmodel, RVmeas):
        return ((np.min(RVmodel) - np.min(RVmeas) ) / self.constGLSRV) **2

    def getErrorAmRefRatio(self, model_instance):
        AmRefRv = model_instance.getScalar('Rv','Patch','Rv1','AmRef')*3
        AmRefSv = model_instance.getScalar('Sv','Patch','Sv1','AmRef')
        return (np.max([ AmRefSv / AmRefRv - self.maxAmRefRatio, 0  ]) / self.constAmRefRatio)**2

    def getCost2StrainTime(self, time, strainMod, strainMeas):

        argmaxMod = np.argmax(strainMod)
        argminMod = np.argmin(strainMod)

        argmaxMeas = np.argmax(strainMeas)
        argminMeas = np.argmin(strainMeas)

        timeMod, strainMod1 = self.StrainTimeStrainManip(time, strainMod)
        timeMeas, strainMeas1 = self.StrainTimeStrainManip(time, strainMeas)

        if False:
            plt.figure()
            plt.plot(strainMod1, timeMod)
            plt.plot(strainMeas1, timeMeas)

        nSteps = 100
        dtMod = np.interp(np.linspace(0.05,0.95,nSteps), strainMod1, timeMod)
        dtMeas = np.interp(np.linspace(0.05,0.95,nSteps), strainMeas1, timeMeas)

        e = np.sum(((dtMod-dtMeas) / self.constStrainTime) ** 2) / nSteps
        return e

    def StrainTimeStrainManip(self, time, strain):
        time = np.array(time)
        strain = np.array(strain)

        argmax = np.argmax(strain)
        argmin = np.argmin(strain)

        if argmin > argmax:
            time = time[argmax:argmin]
            strain = strain[argmax:argmin]
        else:
            time = np.concatenate((time[argmax:]-time[-1], time[1:argmin]))
            strain =np.concatenate((strain[argmax:], strain[1:argmin]))

        strain = strain - np.max(strain)
        strain = strain / np.min(strain)

        return time, np.maximum.accumulate(strain)


    def getErrorK1(self, model_instance):
        e = 0

        k1 = model_instance.getScalar('Lv','Patch','Lv1','k1')
        e+= (np.max([ self.mink1 - k1 , 0 ]) / self.constMink1)**2
        k1 = model_instance.getScalar('Sv','Patch','Sv1','k1')
        e+= (np.max([ self.mink1 - k1 , 0 ]) / self.constMink1)**2
        k1 = model_instance.getScalar('Rv','Patch','Rv1','k1')
        e+= (np.max([ self.mink1 - k1 , 0 ]) / self.constMink1)**2
        k1 = model_instance.getScalar('Rv','Patch','Rv2','k1')
        e+= (np.max([ self.mink1 - k1 , 0 ]) / self.constMink1)**2
        k1 = model_instance.getScalar('Rv','Patch','Rv3','k1')
        e+= (np.max([ self.mink1 - k1 , 0 ]) / self.constMink1)**2

        if e>0:
            e

        return e

    def getErrorSfAct(self, model_instance):
        e = 0

        k1 = model_instance.getScalar('Lv','Patch','Lv1','SfAct')
        e+= (np.max([ self.minSfACt - k1 , 0 ]) / self.constMinSfACt)**2
        k1 = model_instance.getScalar('Sv','Patch','Sv1','SfAct')
        e+= (np.max([ self.minSfACt - k1 , 0 ]) / self.constMinSfACt)**2
        k1 = model_instance.getScalar('Rv','Patch','Rv1','SfAct')
        e+= (np.max([ self.minSfACt - k1 , 0 ]) / self.constMinSfACt)**2
        k1 = model_instance.getScalar('Rv','Patch','Rv2','SfAct')
        e+= (np.max([ self.minSfACt - k1 , 0 ]) / self.constMinSfACt)**2
        k1 = model_instance.getScalar('Rv','Patch','Rv3','SfAct')
        e+= (np.max([ self.minSfACt - k1 , 0 ]) / self.constMinSfACt)**2

        if e>0:
            e
        return e
    
    def getErrorMaxSfAct(self, model_instance):
        e = 0

        SfAct = model_instance.getScalar('Lv','Patch','Lv1','SfAct')
        e+= (np.max([ SfAct - self.maxSfAct, 0 ]) / self.constMaxSfAct)**2
        SfAct = model_instance.getScalar('Sv','Patch','Sv1','SfAct')
        e+= (np.max([ SfAct - self.maxSfAct, 0 ]) / self.constMaxSfAct)**2
        SfAct = model_instance.getScalar('Rv','Patch','Rv1','SfAct')
        e+= (np.max([ SfAct - self.maxSfAct, 0 ]) / self.constMaxSfAct)**2
        SfAct = model_instance.getScalar('Rv','Patch','Rv2','SfAct')
        e+= (np.max([ SfAct - self.maxSfAct, 0 ]) / self.constMaxSfAct)**2
        SfAct = model_instance.getScalar('Rv','Patch','Rv3','SfAct')
        e+= (np.max([ SfAct - self.maxSfAct, 0 ]) / self.constMaxSfAct)**2

        if e>0:
            e
        return e
    
    def getErrorMaxQ0(self, model_instance):
        q0 = model_instance.getScalar('','','','q0')
        return (np.max([q0-self.maxQ0 ,0]) / self.constQ0) **2

     
    def getErrorMinAVdelay(self, model_instance):
        e = 0
        TauAv = model_instance.getScalar('','','','TauAv')
        dTmin = -TauAv*self.minAVfac

        dT = model_instance.getScalar('Lv','Patch','Lv1','dT')
        e+= (np.max([ dTmin - dT, 0 ]) / self.constMinAVdelay)**2
        dT = model_instance.getScalar('Sv','Patch','Sv1','dT')
        e+= (np.max([ dTmin - dT, 0 ]) / self.constMinAVdelay)**2
        dT = model_instance.getScalar('Rv','Patch','Rv1','dT')
        e+= (np.max([ dTmin - dT, 0 ]) / self.constMinAVdelay)**2
        dT = model_instance.getScalar('Rv','Patch','Rv2','dT')
        e+= (np.max([ dTmin - dT, 0 ]) / self.constMinAVdelay)**2
        dT = model_instance.getScalar('Rv','Patch','Rv3','dT')
        e+= (np.max([ dTmin - dT, 0 ]) / self.constMinAVdelay)**2

        if e>0:
            e
        return e
    
    def getErrorMaxDT(self, model_instance):
        e = 0
        tCycle = model_instance.getScalar('','','','tCycle')
        dTmax = tCycle*self.maxDtFac

        dT = model_instance.getScalar('Lv','Patch','Lv1','dT')
        e+= (np.max([ dT - dTmax, 0 ]) / self.constMaxDt)**2
        dT = model_instance.getScalar('Sv','Patch','Sv1','dT')
        e+= (np.max([ dT - dTmax, 0 ]) / self.constMaxDt)**2
        dT = model_instance.getScalar('Rv','Patch','Rv1','dT')
        e+= (np.max([ dT - dTmax, 0 ]) / self.constMaxDt)**2
        dT = model_instance.getScalar('Rv','Patch','Rv2','dT')
        e+= (np.max([ dT - dTmax, 0 ]) / self.constMaxDt)**2
        dT = model_instance.getScalar('Rv','Patch','Rv3','dT')
        e+= (np.max([ dT - dTmax, 0 ]) / self.constMaxDt)**2

        if e>0:
            e
        return e
    


    def getErrorLVdiastolicPositiveStrain(self, strainMod):
        amin = np.argmin(np.mean(strainMod[:,-2:],axis=1))
        strainModSVLV = strainMod[amin:,-2:]
        strainModSVLV[strainModSVLV<0]=0
        e = np.max([0, np.sum(strainModSVLV) - 1])**2 * 10

        return e


    def getErrorNonNegativeVolumes(self,model_instance):
        loc = ['Lv', 'Rv', 'La', 'Ra']
        for i in range(4):
            V = np.array(model_instance.getVector(loc[i], 'Cavity', loc[i], 'V'))
            if np.any(V < 0):
                return np.Infinity
        return 0

    def getErrorLVEF(self,patient,model_instance):
        Vmod = model_instance.getVector('Lv','Cavity','Lv','V')
        LVEFmod = 100 * (np.max(Vmod)-np.min(Vmod) ) / np.max(Vmod)
        if 'LVEF' in patient.scalarData:
            LVEFmeas = patient.scalarData['LVEF']
        elif 'LVEF_Echo' in patient.scalarData:
            LVEFmeas = patient.scalarData['LVEF_Echo']
        return ((LVEFmod-LVEFmeas)/self.constLVEF)**2
    
    def getErrorMaxRVEF(self, model_instance):
        Vmod = model_instance.getVector('Rv','Cavity','Rv','V')
        RVEFmod = 100 * (np.max(Vmod)-np.min(Vmod) ) / np.max(Vmod)
        
        return (np.max([ RVEFmod - self.maxRVEF, 0 ]) / self.constMaxRVEF)**2
    
    def getErrorSeptalBulging(self, model_instance):
        Cm = model_instance.getVector('Sv','Wall','Sv','Cm')
        
        return (np.max([0,self.minCm - np.min(Cm)]))**2/self.constMinCm
    
    def getErrorEDV(self,patient,model_instance):
        Vmod = model_instance.getVector('Lv','Cavity','Lv','V')
        EDVmod = np.max(Vmod)*1e6
        if 'EDV' in patient.scalarData:
            EDVmeas = patient.scalarData['EDV']
        elif 'MRI_LVEDV' in patient.scalarData:
            EDVmeas = patient.scalarData['MRI_LVEDV']
        return ((EDVmod-EDVmeas)/self.constEDV)**2
    
    def getErrorRVD(self,patient,model_instance):
        Am_Sv = np.array(model_instance.getVector('Sv','Wall','Sv','Am'))
        Am_Rv = np.array(model_instance.getVector('Rv','Wall','Rv','Am'))
        
        VWall_Sv = model_instance.getScalar('Sv','Wall','Sv','VWall')
        VWall_Rv = model_instance.getScalar('Rv','Wall','Rv','VWall')
        Ym = np.array(model_instance.getVector('Sv','TriSeg','Sv','Y'))
        
        signCm_Sv = np.sign(model_instance.getVector('Sv','Wall','Sv','Cm'))
        signCm_Rv = np.sign(model_instance.getVector('Rv','Wall','Rv','Cm'))
        
        Xm_Sv = signCm_Sv * np.sqrt(Am_Sv/np.pi - Ym**2) #+ VWall_Sv/Am_Sv / 2
        Xm_Rv = signCm_Rv * np.sqrt(Am_Rv/np.pi - Ym**2) #- VWall_Rv/Am_Rv/ 2
        
        RVDmod = np.max(Xm_Rv-Xm_Sv)*1e3
        RVDmeas = patient.scalarData['RVD']
        return ((RVDmod-RVDmeas)/self.constRVD)**2

    def getErrorMLAP(self,patient,model_instance):
        p = np.mean(model_instance.getVector('La','Node','La','p'))
        return ((np.max([0, p-self.limMaxMLAP]))/self.constMaxMLAP)**2

    def getErrorMRAP(self,patient,model_instance):
        p = np.mean(model_instance.getVector('Ra','Node','La','p'))
        return ((np.max([0, p-self.limMaxMRAP]))/self.constMaxMRAP)**2

    def getCost2SingleStrain(self,time,strainMod,strainMeas,const=1, rateconst =0 ):
        e_strain = np.sum( (strainMod-strainMeas)**2 ) / len(strainMod) * time[-1]
        e_rate = 0
        if rateconst>0:
            if len(strainMod)>1: # these simulations will have inifint error due to other indices
                minMeas = np.max(strainMeas) - np.min(strainMeas)
                minMod = np.max(strainMod) - np.min(strainMod)
                
                rateMeas = np.diff(strainMeas/minMeas)
                rateMod = np.diff(strainMod/minMod)
                
                if self.useMaxRate:
                    rateMeas[rateMeas>self.maxRate] = self.maxRate 
                    rateMeas[rateMeas<-self.maxRate] = -self.maxRate
                    rateMod[rateMod>self.maxRate] = self.maxRate 
                    rateMod[rateMod<-self.maxRate] = -self.maxRate 
                    
                e_rate = rateconst * np.sum( (minMeas * (rateMod-rateMeas))**2 ) / (len(strainMod)-1) * time[-2]
        return const*e_strain + rateconst*e_rate

    def getCost2SingleStrainDiff(self,time,strainMod,strainMeas ):
        e_strain = np.sum( (np.diff(strainMod)-np.diff(strainMeas))**2 ) / len(strainMod) * time[-1]
        return e_strain


    def getModelStrainIDX0corrected(self,model_instance,patient=[],timeMeas=[],strainMeas=[]):
        if strainMeas==[]:
            strainMeas = self.getStrainMeas(patient)
        if timeMeas==[]:
            timeMeas = patient.RVtime
        strainMod  = self.getStrainModel(model_instance=model_instance,idx0='onsetQRS')
        if np.any(np.isnan(strainMod)):
            return np.nan, [np.nan, np.nan]
        timeMod    = model_instance.getTime()

        if self.splitRVLV:
            # RV
            strainMod[:,:3], t0RV = self.getModelStrainIDX0correctedSingle(timeMeas,strainMeas[:,:3],timeMod,strainMod[:,:3])
            # LV
            strainMod[:,3:5], t0LV = self.getModelStrainIDX0correctedSingle(timeMeas,strainMeas[:,3:5],timeMod,strainMod[:,3:5])
        else:
            strainMod[:,:5], t0LV = self.getModelStrainIDX0correctedSingle(timeMeas,strainMeas[:,:5],timeMod,strainMod[:,:5])
            t0RV = t0LV

        return strainMod, [t0RV, t0LV]

    def getModelStrainIDX0correctedSingle(self,timeMeas,strainMeas,timeMod,strainMod):
        tAtrialMeas = self.findTimeOnsetAtrialKick(timeMeas, strainMeas, maxTime = 0.1765 * timeMod[-1] * 1.1)
        tAtrialMod = self.findTimeOnsetAtrialKick(timeMod, strainMod, maxTime = 0.1765 * timeMod[-1] * 1.1)
        
        dIDX = int((tAtrialMod - tAtrialMeas)/0.002)

        idxMeas = np.argwhere(tAtrialMeas<np.array(timeMeas))[0,0]
        idxMod = np.argwhere(tAtrialMod<np.array(timeMod))[0,0]

        # interval wrt estimation to test
        intervalIDX = range(-15,15) # +/- 30 ms
        
        segment_weights = np.ones(strainMeas.shape[1])
        if len(segment_weights)==3: # correct for bad RV segments
            strain_n_above_zero = np.sum(strainMeas[idxMeas:,:]>0, axis=0)
            strain_fac_above_zero = strain_n_above_zero / strainMeas[idxMeas:,:].shape[0]
            segment_weights = 1 - strain_fac_above_zero
        segment_weights = segment_weights / np.sum(segment_weights)

        onsetStrainMeas = np.sum(segment_weights * np.concatenate((strainMeas[idxMeas:,:],strainMeas[1:(idxMeas+1),:])),axis=1)
        idxStartMeasurement = len(strainMeas) - idxMeas

        meanStrainMod = np.sum(segment_weights * strainMod,axis=1)

        #idxMeasEnd = np.argwhere(
        #    (np.array(range(len(onsetStrainMeas)))> np.argmax(onsetStrainMeas)) &
        #    (onsetStrainMeas < 0.05*np.min(onsetStrainMeas))
        #)[0,0]

        #idxMeasEnd = np.argmax(timeMeas>= (timeMod[-1]-tAtrialMeas + 0.05)  ) + 1
        idxMeasEnd = np.argmax(onsetStrainMeas)
        
        idxModEnd = np.argwhere(timeMod>=timeMeas[idxMeasEnd])[0,0]

        onsetStrainMeas = (onsetStrainMeas-onsetStrainMeas[0])
        onsetStrainMeas = onsetStrainMeas[:idxMeasEnd]/onsetStrainMeas[-idxMeas]
        onsetStrainMeasDiff = np.diff(onsetStrainMeas)

        iIDXmin = 0
        eMin = np.Infinity

        for iIDX in intervalIDX:
            onsetStrainMod = np.concatenate((meanStrainMod[(idxMod+iIDX):],meanStrainMod[1:(idxMod+iIDX+1)]))
            if -(idxMod+iIDX) < len(onsetStrainMod):
                onsetStrainMod = (onsetStrainMod-onsetStrainMod[0])
                onsetStrainMod = onsetStrainMod / onsetStrainMod[-(idxMod+iIDX)]

                onsetStrainModDisc = np.interp(timeMeas[:(idxMeasEnd)], timeMod[:(idxModEnd)], onsetStrainMod[:idxModEnd])

                e = np.sum((onsetStrainMeasDiff - np.diff(onsetStrainModDisc))**2)
                #e = np.sum((onsetStrainMeas - onsetStrainModDisc)**2)

                if e<eMin:
                    eMin=e
                    iIDXmin = iIDX

                if False:
                    plt.figure(1)
                    plt.subplot(3,1,1)
                    plt.plot(onsetStrainMeasDiff)
                    plt.plot(np.diff(onsetStrainModDisc))
                    plt.subplot(3,1,2)
                    plt.plot(onsetStrainMeas[:idxMeasEnd])
                    plt.plot(onsetStrainModDisc)
                    plt.title('e: ' + str(e))
                    plt.subplot(3,1,3)
                    plt.scatter(iIDX, e)

        X = (np.array(range(strainMod.shape[0]))  + dIDX + iIDXmin).astype(int)
        X[X<0] = X[X<0] - 1 # first point is always same as last point
        X[X>=len(X)] = X[X>=len(X)] - len(X)
        strainMod[:,:] = strainMod / 100 + 1
        strainMod[:,:] = strainMod[X,:]
        strainMod[:,:] = (strainMod[:,:] / strainMod[0,:] - 1) * 100

        if self.normalize_modelled_strain:
            strainModMin = np.min(meanStrainMod)
            strainMeasMin = np.min(np.mean(strainMeas,axis=1))
            strainMod = strainMod / strainModMin * strainMeasMin

        return strainMod, timeMod[X[0]]

    def getDiscretizedStrain(self,modelTime, modelStrain, measurementTime):
        modStrainDiscretized = np.ndarray((len(measurementTime),modelStrain.shape[1]))
        for iSeg in range(modelStrain.shape[1]):
            modStrainDiscretized[:,iSeg] = np.interp(measurementTime, modelTime, modelStrain[:,iSeg])
        return modStrainDiscretized

    def plotValidation(self,patient=[],model_instance=[],vectorData=[]):
        strainMeas = self.getStrainMeas(patient)
        timeMeas   = patient.RVtime
        strainMod,t0  = self.getModelStrainIDX0corrected(model_instance=model_instance,strainMeas=strainMeas,timeMeas=timeMeas)
        timeMod    = model_instance.getTime()

        print(t0)

        strainMod = self.getDiscretizedStrain(timeMod,strainMod,timeMeas)
        timeMod=timeMeas
        X = 0
        for iSeg in range(5):
            X = np.max([X,  np.argmin(strainMeas[:,iSeg]) + int(0.05/(timeMod[1]-timeMod[0])) ])
        

        plt.figure()
        plt.subplot(4,3,1)
        X=np.argmin(strainMeas[:,0]) + int(0.05/(timeMod[1]-timeMod[0]))
        self.plotValidationSingleStrain(timeMeas,strainMeas[:,0], timeMod,strainMod[:,0],strainMeas,strainMod, t0[1], X=X)
        plt.subplot(4,3,2)
        X=np.argmin(strainMeas[:,1]) + int(0.05/(timeMod[1]-timeMod[0]))
        self.plotValidationSingleStrain(timeMeas,strainMeas[:,1], timeMod,strainMod[:,1],strainMeas,strainMod, t0[1], X=X)
        plt.subplot(4,3,3)
        X=np.argmin(strainMeas[:,2]) + int(0.05/(timeMod[1]-timeMod[0]))
        self.plotValidationSingleStrain(timeMeas,strainMeas[:,2], timeMod,strainMod[:,2],strainMeas,strainMod, t0[1], X=X)
        plt.subplot(4,3,4)
        X=np.argmin(strainMeas[:,3]) + int(0.05/(timeMod[1]-timeMod[0]))
        self.plotValidationSingleStrain(timeMeas,strainMeas[:,3], timeMod,strainMod[:,3],strainMeas,strainMod, t0[0], X=X)
        plt.subplot(4,3,5)
        X=np.argmin(strainMeas[:,4]) + int(0.05/(timeMod[1]-timeMod[0]))
        self.plotValidationSingleStrain(timeMeas,strainMeas[:,4], timeMod,strainMod[:,4],strainMeas,strainMod, t0[0], X=X)

        minMod = np.max(strainMod, axis=0) - np.min(strainMod, axis=0)
        minMeas = np.max(strainMeas, axis=0) - np.min(strainMeas, axis=0)
                
        minMod[minMod>-1] = -1
        minMeas[minMeas>-1] = -1
        
        print(minMeas)
        for i in range(5):
            plt.subplot(4,3,7+i)
            self.plotValidationSingleStrainRate(timeMeas,strainMeas[:,i], timeMod,strainMod[:,i]/minMod[i]*minMeas[i],strainMeas,strainMod, X=X)

        plt.subplot(4,3,12)
        plt.title('e={:02.2f}'.format(self.getCost2(patient=patient,model_instance=model_instance)))
        plt.xlim([0,100])
        plt.ylim([0,100])
        if self.useLVEF:
            e = self.getErrorLVEF(patient=patient,model_instance=model_instance)
            print(e)
            plt.text(10,80,'eLVEF={:0.2f}'.format(e))
        if self.useEDV:
            e = self.getErrorEDV(patient=patient,model_instance=model_instance)
            print(e)
            plt.text(10,60,'EDV={:0.2f}'.format(e))
        if self.useMaxRVEF:
            e = self.getErrorMaxRVEF(model_instance=model_instance)
            print(e)
            plt.text(10,40,'eRVEF={:0.2f}'.format(e))
        if self.useNoBulging:
            e = self.getErrorSeptalBulging(model_instance=model_instance)
            print(e)
            plt.text(10,20,'eBulg={:0.2f}'.format(e))
        if self.useNoBulging:
            e = self.getErrorMLAP(patient=patient,model_instance=model_instance)
            e1 = self.getErrorMRAP(patient=patient, model_instance=model_instance)
            plt.text(10,0,'eMLAP,eRLAP={:0.2f},{:0.2f}'.format(e, e1))

    def plotValidationSingleStrain(self,timeMeas,strainMeas, timeMod,strainMod,allStrainMeas,allStrainMod, t0=0, X=0):
        tAtrialMeas = self.findTimeOnsetAtrialKick(timeMeas, allStrainMeas)
        tAtrialMod = self.findTimeOnsetAtrialKick(timeMod, allStrainMod)

        plt.plot( timeMod , strainMod )
        plt.plot( timeMeas,strainMeas )

        #X = np.array(range(len(timeMod)))

        meanStrain = np.mean(allStrainMeas[:,3:],axis=1)
        #X = np.argwhere(  (X>np.argmin(meanStrain)) & (meanStrain > 0.75*np.min(meanStrain))  )[0,0]
        #X = np.argmin(strainMeas) + int(0.1/(timeMod[1]-timeMod[0]))
        plt.plot(timeMod[:X],np.ones(X))

        plt.title('e={:0.2f} + {:0.2f}'.format(self.getCost2SingleStrain(timeMod[:X],strainMod[:X],strainMeas[:X], const=self.strainFac[0],rateconst=0),
                                     self.getCost2SingleStrain(timeMod[:],strainMod[:],strainMeas[:], const=self.strainFacFull[0],rateconst=0)))

        plt.scatter(tAtrialMod,0)
        plt.scatter(tAtrialMeas,0)
        
        plt.plot([t0,t0], [np.min(strainMeas), np.max(strainMeas)])

    def plotValidationSingleStrainRate(self,timeMeas,strainMeas, timeMod,strainMod,allStrainMeas,allStrainMod,X=0):
        tAtrialMeas = self.findTimeOnsetAtrialKick(timeMeas, allStrainMeas)
        tAtrialMod = self.findTimeOnsetAtrialKick(timeMod, allStrainMod)

        plt.plot( timeMod[:-1] , np.diff(strainMod) )
        plt.plot( timeMeas[:-1], np.diff(strainMeas) )

        #X = np.array(range(len(timeMod)))

        meanStrain = np.mean(allStrainMeas[:,3:],axis=1)
        #X = np.argwhere(  (X>np.argmin(meanStrain)) & (meanStrain > 0.75*np.min(meanStrain))  )[0,0]
        plt.plot(timeMod[:X],np.ones(X))

        e = self.getCost2SingleStrain(timeMod[:X],strainMod[:X],strainMeas[:X],const=0, rateconst=self.rateconst[0])
        e2 = self.getCost2SingleStrain(timeMod[:],strainMod[:],strainMeas[:],const=0, rateconst=self.rateconstFull[0])
        #e = e - self.getCost2SingleStrain(timeMod[:X],strainMod[:X],strainMeas[:X],rateconst=0)
        plt.title('e={:0.2f} + {:0.2f}'.format(e, e2))

        plt.scatter(tAtrialMod,0)
        plt.scatter(tAtrialMeas,0)
