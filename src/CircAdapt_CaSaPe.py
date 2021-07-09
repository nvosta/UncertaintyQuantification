######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# CircAdapt with manipulation functions
######

import sys
sys.path.insert(1, 'E:/users/nick/Documents/_Nick/programming/cpp/projects/CircAdapt/python3/core/')
from CircAdapt import CircAdapt
import numpy as np

class CircAdapt_CaSaPe(CircAdapt):
    computedData = {}

    def run(self):
        '''
            Run model, used in CaSaPe
        '''
        self.setScalar('','','','minTol',1e-3 )
        self.setScalar('','','','maxBeats',100)
        self.setScalar('','','','dt',0.002)
        self.runStable()
        self.computedData = {}

    def runHQ(self):
        '''
            Run model, used in CaSaPe
        '''
        minTol = 1e-3
        self.setScalar('','','','minTol',1e-4 )
        self.setScalar('','','','solverTol',1e-5 )
        self.setScalar('','','','maxBeats',100)
        self.setScalar('','','','dt',0.002)
        self.runStable()
        self.setScalar('','','','minTol',minTol )
        self.computedData = {}

    def runSHQ(self):
        '''
            Run model, used in CaSaPe
        '''
        minTol = 1e-3
        self.setScalar('','','','minTol',1e-5 )
        self.setScalar('','','','solverTol',1e-5 )
        self.setScalar('','','','maxBeats',100)
        self.setScalar('','','','dt',0.002)
        self.runStable()
        self.setScalar('','','','minTol',minTol )
        self.computedData = {}

    def runFast(self):
        '''
            Run model, used in CaSaPe
        '''
        minTol = 1e-3
        self.setScalar('','','','minTol',1e-2 )
        self.setScalar('','','','maxBeats',100)
        self.setScalar('','','','dt',0.002)
        self.runStable()
        self.setScalar('','','','minTol', minTol )
        self.computedData = {}


    def getStrain(self,loc,t0IDX='onsetQRS'):
        '''
            Get Strain
        '''
        if t0IDX=='onsetQRS':
            t0IDX = self.onsetQRS()

        # get Time
        if type(loc)==list:
            tCycle = self.getScalar('','','','tCycle')
            strain = np.ndarray((int(round(tCycle/0.002)+1),len(loc)))
            for iL in range(len(loc)):
                strain[:,iL] = self.getStrain(loc[iL],t0IDX)
        else:
            strain = np.array(self.getVector(loc[0:2],'Patch',loc,'Ls'))
            # todo: improve circadapt dll
            tCycle = self.getScalar('','','','tCycle')
            time = self.getTime()
            
            strain = np.interp(np.linspace(0, tCycle, int(round(tCycle/0.002)+1)),
                               time, strain)


            if t0IDX==-1:
                t0IDX=0
            strain = np.concatenate((strain[t0IDX:-1], strain[:(t0IDX+1)]))
            strain = (strain / strain[0]-1) * 100
           
        return strain

    def getStretch(self,loc,t0IDX):
            '''
                Get Strain
            '''
            t0IDX=max(t0IDX,0)
            # get Time
            if type(loc)==list:
                stretch = np.ndarray((self.CA.getVecLength(),len(loc)))
                for iL in range(len(loc)):
                    stretch[:,iL] = self.getStretch(loc[iL],t0IDX)
            else:
                if loc=='GL':
                    stretch1 = np.array(self.getVector(loc[0:2],'Patch','Sv1','Ls'))
                    stretch2 = np.array(self.getVector(loc[0:2],'Patch','Lv1','Ls'))
                    VWall1 = self.getScalar('Sv','Wall','','VWall')
                    VWall2 = self.getScalar('Lv','Wall','','VWall')
                    stretch = (stretch1*VWall1 + stretch2*VWall2) / (VWall1 + VWall2)
                else:
                    stretch = np.array(self.getVector(loc[0:2],'Patch',loc,'Ls'))
                stretch = np.concatenate((stretch[t0IDX:-1], stretch[:(t0IDX+1)]))
                stretch = stretch / stretch[0]
            return stretch

    def getTime(self):
        if 'Time' in self.computedData:
            return self.computedData['Time']
        self.computedData['Time'] = self.getVector('Time')
        return self.computedData['Time']

    def getComputedData(self,keyToGive,func,*args, **kwargs):
        if keyToGive in self.computedData:
            return self.computedData[keyToGive]
        # compute data
        res = func(*args,**kwargs)
        self.computedData[keyToGive] = res
        return res

    def onsetQRS(self):
        if 'onsetQRS' in self.computedData:
            return self.computedData['onsetQRS']

        tCycle = self.getScalar('','','','tCycle')
        CLV = np.array(self.getVector('Lv','Patch','Lv1','C'))
        CRV = np.array(self.getVector('Lv','Patch','Rv1','C'))
        CSV = np.array(self.getVector('Lv','Patch','Sv1','C'))
        CL = CLV+CRV+CSV

        CLV = np.array(self.getVector('Lv','Patch','Lv1','CDot'))
        CRV = np.array(self.getVector('Lv','Patch','Rv1','CDot'))
        CSV = np.array(self.getVector('Lv','Patch','Sv1','CDot'))
        CDotL = CLV+CRV+CSV

        CMax = max(CL)
        idxCMax = np.argmax(CL)

        #P  = self.getPdict()

        t = np.array(self.getVector('','','','Time'))

        tCLmax = t[idxCMax]

        boolRgL = (0.01*CMax < CL) & (CL < 0.3*CMax) & (CDotL>0)
        RgL = np.argwhere( boolRgL )



        #print(RgL,boolRgL,t,self.getVector('','','','t'))
        tRgL = t[boolRgL]
        ML = np.zeros((len(tRgL),2))
        ML[:,1] = tRgL

        #print(ML)

        if len(tRgL)==0:
            np.save('Pcrash_tRgL0.npy',self.getPdictCompressed(),allow_pickle=True)
            #print('  - tRgL0')

            ## Todo!!!
            return np.nan

        derivative = (0.3*CMax -0.01*CMax) / len(tRgL)

        # SUCCESS
        if sum(boolRgL)>0:
            onQRS = int(np.argwhere(boolRgL)[0] - np.ceil(0.01*CMax / derivative))
            self.computedData['onsetQRS'] = onQRS
            return onQRS

        # FAILED::

        P = self.getPdict()

        np.save('Pcrash.npy',P,allow_pickle=True)

        ## Todo!!!
        return 25
