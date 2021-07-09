######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# Patient Dataset
######

import numpy as np
import os.path

######
# Patient
######

class Patient:
    '''
        Patient: Contains Patient strain and scalar data
    '''
    def __init__(self, id=0, virtualModel=[], filename=[],observation=[],patientDict=[]):
        self.id = str(id)
        if not(observation==[]):
            self.id = self.id + '_' + str(observation)
        self.patientID = id
        self.observation = observation

        self.RVtime   = []
        self.RVstrain = []
        self.LVstrain = []
        self.SVstrain = []
        self.GLstrain = []
        self.GRstrain = []
        self.nLV = 0
        self.nSV = 0

        self.LV2time   = []
        self.LV2strain = []

        self.LV3time   = []
        self.LV3strain = []

        self.LV4time   = []
        self.LV4strain = []

        self.strainRaw = {}

        self.scalarData = []

        self.tCycle = []

        self.modelPdict = []

        self.computedData = {}

        if not(patientDict==[]):
            self.loadPatient(patientDict=patientDict)
        elif not(filename==[]):
            self.loadPatient(filename=filename)
        elif not(virtualModel==[]):
            self.loadStrainFromModel(virtualModel)
            self.loadScalarFromModel(virtualModel)

    def savePatient(self,folder,filename=[]):
        if filename==[] and type(self.modelPdict)==list and self.modelPdict==[]:
            filename = 'pat' + self.id + '.npy'
        elif filename==[]:
            filename = '' + self.id + '.npy'
        data = {'id':self.id,
                'patientID':self.patientID,
                'observation':self.observation,
                'RVtime':     self.RVtime,
                'RVstrain':   self.RVstrain,
                'LVstrain':   self.LVstrain,
                'SVstrain':   self.SVstrain,
                'GLstrain':   self.GLstrain,
                'GRstrain':   self.GRstrain,
                'nLV':        self.nLV,
                'nSV':        self.nSV,
                'LV2time':    self.LV2time,
                'LV2strain':  self.LV2strain,
                'LV3time':    self.LV3time,
                'LV3strain':  self.LV3strain,
                'LV4time':    self.LV4time,
                'LV4strain':  self.LV4strain,
                'strainRaw':  self.strainRaw,
                'scalarData': self.scalarData,
                'tCycle':     self.tCycle,
                'modelPdict':self.modelPdict}
        print('Save data to ', folder+filename)
        np.save(folder+filename,data,allow_pickle=True)

    def loadPatient(self,filename=[],patientDict=[]):
        if patientDict==[]:
            data = np.load(filename,allow_pickle=True)
            data = data.item()
        else:
            data=patientDict
        self.id         = data['id']
        self.observation= data['observation']
        self.patientID  = data['patientID']
        self.RVtime     = data['RVtime']
        self.RVstrain   = data['RVstrain']
        self.LVstrain   = data['LVstrain']
        self.SVstrain   = data['SVstrain']
        self.GLstrain   = data['GLstrain']
        self.GRstrain   = data['GRstrain']
        self.nLV        = data['nLV']
        self.nSV        = data['nSV']
        self.LV2time    = data['LV2time']
        self.LV2strain  = data['LV2strain']
        self.LV3time    = data['LV3time']
        self.LV3strain  = data['LV3strain']
        self.LV4time    = data['LV4time']
        self.LV4strain  = data['LV4strain']
        self.strainRaw  = data['strainRaw']
        self.scalarData = data['scalarData']
        self.tCycle     = data['tCycle']
        self.modelPdict = data['modelPdict']


    def loadRVstrain(self,RVstrainFile):
        RVstrainRaw   = self.loadStrain(RVstrainFile)
        RVstrainRaw   = self.stripStrain(RVstrainRaw)
        self.strainRaw['RV'] = RVstrainRaw

        self.RVtime   = RVstrainRaw['time']
        self.RVstrain = RVstrainRaw['strain']

        self.tCycle = np.max(self.RVtime)

        self.GRstrain = np.mean(RVstrainRaw['strain'],1)

    def loadLV2strain(self,LV2strainFile):
        if os.path.isfile(LV2strainFile):
            LVstrainRaw   = self.loadStrain(LV2strainFile)
            LVstrainRaw   = self.stripStrain(LVstrainRaw)
            self.strainRaw['LV2'] = LVstrainRaw

            self.LV2time   = LVstrainRaw['time']
            self.LV2strain = LVstrainRaw['strain']
        else:
            self.strainRaw['LV2']=[]
            self.LV2time=[]
            self.LV2strain=[]

    def loadLV3strain(self,LV3strainFile):
        if os.path.isfile(LV3strainFile):
            LVstrainRaw   = self.loadStrain(LV3strainFile)
            LVstrainRaw   = self.stripStrain(LVstrainRaw)
            self.strainRaw['LV3'] = LVstrainRaw

            self.LV3time   = LVstrainRaw['time']
            self.LV3strain = LVstrainRaw['strain']
        else:
            self.strainRaw['LV3']=[]
            self.LV3time=[]
            self.LV3strain=[]

    def loadLV4strain(self,LV4strainFile):
        if os.path.isfile(LV4strainFile):
            LVstrainRaw   = self.loadStrain(LV4strainFile)
            LVstrainRaw   = self.stripStrain(LVstrainRaw)
            self.strainRaw['LV4'] = LVstrainRaw

            self.LV4time   = LVstrainRaw['time']
            self.LV4strain = LVstrainRaw['strain']
        else:
            self.strainRaw['LV4']=[]
            self.LV4time=[]
            self.LV4strain=[]

    def loadStrain(self,filename):
        # init return
        strainRaw = {}

        # load strain file
        text_file = open(filename, "r")
        try:
            lines = text_file.readlines()

            # iterate over lines
            for iLine in range(len(lines)):
                line = lines[iLine]
                if iLine==0:
                    if not(line[0:16]==  'Segmental Traces'):
                        strainRaw['error']=2
                        return strainRaw
                elif iLine==1:
                    if not(line[0:17]=='Number of Frames '):
                        strainRaw['error']=2
                        return strainRaw
                    else:
                        strainRaw['nFrames'] = int(line.replace('Number of Frames ',''))
                elif iLine==2:
                    #FR=110 Left Marker Time=0.219000 Right Marker Time=1.312000 ES Time=0.647000
                    if not(line[0:3]=='FR='):
                        strainRaw['error']=1
                        return strainRaw
                    else:
                        spl = line.replace('FR=','').strip().split(' Left Marker Time=')
                        line = spl[1]
                        strainRaw['FR'] = int(spl[0])
                        spl = line.replace('FR=','').split(' Right Marker Time=')
                        line = spl[1]
                        strainRaw['LMT'] = float(spl[0])
                        spl = line.replace('FR=','').split(' ES Time=')
                        strainRaw['RMT'] = float(spl[0])
                        strainRaw['ES'] = float(spl[1])
                elif iLine==3:
                    spl = line.strip().split('\t')
                    if not(spl[0]=='Time (s) '):
                        strainRaw['error']=2

                    isECG = spl[-1]=='ECG :'
                    isEmpty2 = spl[1]==''

                    if isECG:
                        spl= spl[:-1] # correct for ECG
                        strainRaw['ECG'] = np.ndarray(strainRaw['nFrames'])
                    if isEmpty2:
                        spl= spl[1:] # correct for empty
                    spl = spl[1:] # correct for time

                    for iS in range(len(spl)):
                        spl[iS] = spl[iS].strip()
                    strainRaw['segments'] = spl

                    # add strain and time
                    strainRaw['strain'] = np.ndarray((strainRaw['nFrames'] ,len(spl)))
                    strainRaw['time'] = np.ndarray(strainRaw['nFrames'])

                    isEmpty2 = True
                else:
                    spl = line.strip().split('\t')
                    strainRaw['time'][iLine-4] = float(spl[0])
                    if isECG:
                        strainRaw['ECG'][iLine-4] = float(spl[-1])
                    for iSeg in range(len(strainRaw['segments'])):
                        strainRaw['strain'][iLine-4,iSeg] = float(spl[1+isEmpty2+iSeg])
        except:
            strainRaw['error']=1
        finally:
            text_file.close()
        return strainRaw

    def stripStrain(self,strainRaw):
        '''
            Strip first and lasts iters from strain outside beat
        '''

        # find idx
        idx = [ n for n,i in enumerate(strainRaw['time']) if i>=strainRaw['LMT'] and i<=strainRaw['RMT'] ]

        # strip time, strain, ecg
        strainRaw['time']=strainRaw['time'][idx] - strainRaw['time'][idx[0]]
        strainRaw['strain']=strainRaw['strain'][idx,:]
        if 'ECG' in strainRaw.keys():
            strainRaw['ECG']=strainRaw['ECG'][idx]

        # recalculate end systole, left and right marker time, and nFrames
        strainRaw['ES'] = strainRaw['ES'] - strainRaw['LMT']
        strainRaw['LMT'] = strainRaw['time'][0]
        strainRaw['RMT'] = strainRaw['time'][-1]
        strainRaw['nFrames'] = len( strainRaw['time'])
        return strainRaw

    def calculateLVandIVS(self):
        # Calculate LV free wall and Inter ventricular septum strain out of LV2,LV3,LV4
        if not('RV' in self.strainRaw) or not('LV2' in self.strainRaw)  or not('LV3' in self.strainRaw)  or not('LV4' in self.strainRaw) :
            return False

        nLV = 0
        nSV = 0

        LVseg = {'basInf','midInf','apInf','apAnt','midAnt','basAnt','basPost','midPost','apPost','apLat','midLat','basLat'}
        SVseg = {'apAntSept','midAntSept','basAntSept','basSept','midSept','apSept'}

        LVstrain = self.RVtime * 0
        SVstrain = self.RVtime * 0

        import matplotlib.pyplot as plt


        useDataSets = ['LV2','LV3','LV4']
        for iD in range(len(useDataSets)):
            if not self.strainRaw[useDataSets[iD]]==[]:
                for iSeg in range(len(self.strainRaw[useDataSets[iD]]['segments'])):

                    s = np.interp(self.RVtime, self.strainRaw[useDataSets[iD]]['time'], self.strainRaw[useDataSets[iD]]['strain'][:,iSeg])

                    s1 = np.interp(self.RVtime, self.strainRaw[useDataSets[iD]]['time'] - self.strainRaw[useDataSets[iD]]['time'][-1] + self.RVtime[-1], self.strainRaw[useDataSets[iD]]['strain'][:,iSeg])

                    dT = self.RVtime[1] - self.RVtime[0]
                    useTime = np.max([0.1*self.RVtime[-1], self.strainRaw[useDataSets[iD]]['time'][-1] - self.RVtime[-1]])
                    useTime = 0.300

                    nIterAtrialKick = int(np.ceil(useTime/dT))

                    halfway = int(len(s)/2)

                    k = 100
                    x0 = self.RVtime[-1]-useTime
                    a=1

                    fac = (1 / (1+np.exp(-k*(self.RVtime[halfway:]-x0)))**a)

                    s[halfway:] =   (1-fac)*s[halfway:] +  fac*s1[halfway:]

                    #s[-nIterAtrialKick:] = s1[-nIterAtrialKick:]




                    if self.strainRaw[useDataSets[iD]]['segments'][iSeg] in LVseg:
                        nLV = nLV + 1
                        LVstrain = LVstrain + s
                    elif self.strainRaw[useDataSets[iD]]['segments'][iSeg] in SVseg:
                        nSV = nSV + 1
                        SVstrain = SVstrain + s
                    else:
                        print(self.strainRaw[useDataSets[iD]]['segments'][iSeg] + ' not included in IVS/LV')


        self.GLstrain = (LVstrain + SVstrain) / (nLV+nSV)
        self.LVstrain = LVstrain / nLV
        self.SVstrain = SVstrain / nSV

        self.nLV = nLV
        self.nSV = nSV

    def matchECG(self):

        # Find RV max ecg
        RVmaxECG = np.argmin(self.strainRaw['RV']['ECG'][:10])

        if False:
            LVmaxECG = np.argmin(self.strainRaw['LV2']['ECG'][:10])
            import matplotlib.pyplot as plt
            print(self.strainRaw.keys())
            print(self.strainRaw['RV'].keys())

            plt.figure()
            plt.subplot(421)
            plt.plot(self.strainRaw['RV']['time'],self.strainRaw['RV']['ECG'])
            plt.scatter(self.strainRaw['RV']['time'][RVmaxECG], self.strainRaw['RV']['ECG'][RVmaxECG])
            plt.subplot(423)
            plt.plot(self.strainRaw['RV']['time'],self.strainRaw['RV']['strain'])


            plt.subplot(422)
            plt.plot(self.strainRaw['LV2']['time'],self.strainRaw['LV2']['ECG'])
            plt.scatter(self.strainRaw['LV2']['time'][LVmaxECG], self.strainRaw['LV2']['ECG'][LVmaxECG])
            plt.subplot(424)
            plt.plot(self.strainRaw['LV2']['time'],self.strainRaw['LV2']['strain'])

        # Match peak ECG LV with RV
        nItersEnvelope=10
        ECGrvEnvelope = self.strainRaw['RV']['ECG'][:nItersEnvelope]
        for LVloc in {'LV2','LV3','LV4'}:
            print('strain: ',self.strainRaw[LVloc]['strain'])
            print('strain: ',self.strainRaw[LVloc]['strain'][0,:])



            LVmaxECG = np.argmin(self.strainRaw[LVloc]['ECG'][:10])
            shift = int(np.round((self.strainRaw[LVloc]['time'][LVmaxECG]-self.strainRaw['RV']['time'][RVmaxECG]) / self.strainRaw[LVloc]['time'][1]))

            if shift<0:
                shift = shift + len(self.strainRaw[LVloc]['ECG'])

            self.strainRaw[LVloc]['ECG'] = np.concatenate((self.strainRaw[LVloc]['ECG'][shift:-1], self.strainRaw[LVloc]['ECG'][:(shift+1)]))

            self.strainRaw[LVloc]['strain'] = (self.strainRaw[LVloc]['strain']/100)+1
            self.strainRaw[LVloc]['strain'] = np.concatenate((self.strainRaw[LVloc]['strain'][shift:-1,:], self.strainRaw[LVloc]['strain'][:(shift+1),:]))
            self.strainRaw[LVloc]['strain'] = self.strainRaw[LVloc]['strain'] / self.strainRaw[LVloc]['strain'][0,:]
            self.strainRaw[LVloc]['strain'] = (self.strainRaw[LVloc]['strain']-1)*100

        if False:
            plt.subplot(426)
            plt.plot(self.strainRaw['LV2']['time'],self.strainRaw['LV2']['ECG'])
            plt.scatter(self.strainRaw['LV2']['time'][RVmaxECG], self.strainRaw['LV2']['ECG'][RVmaxECG])
            plt.subplot(428)
            plt.plot(self.strainRaw['LV2']['strain'])

            plt.show()

    def setScalarData(self,scalarData):
        self.scalarData = scalarData

    def loadStrainFromModel(self,model_instance):
        self.nLV = 2
        self.nSV = 1

        strain = np.array(model_instance.getStrain(['Rv1','Rv2','Rv3','Lv1','Sv1']))
        time = np.array(model_instance.getVector('','','','Time'))

        self.RVtime   = time

        self.RVstrain = strain[:,:3]
        self.LVstrain = strain[:,3]
        self.SVstrain = strain[:,4]
        self.GLstrain = (strain[:,3]*2+strain[:,4])/3
        self.GRstrain = np.mean(strain[:,:3],1)

        self.tCycle = np.max(self.RVtime)

        self.modelPdict = model_instance.getPdictCompressed()

    def loadScalarFromModel(self,model_instance):
        Vmod = model_instance.getVector('Lv','Cavity','Lv','V')
        LVEFmod = 100 * (np.max(Vmod)-np.min(Vmod) ) / np.max(Vmod)
        
        
        # calculate RVD
        Am_Sv = np.array(model_instance.getVector('Sv','Wall','Sv','Am'))
        Am_Rv = np.array(model_instance.getVector('Rv','Wall','Rv','Am'))
        
        VWall_Sv = model_instance.getScalar('Sv','Wall','Sv','VWall')
        VWall_Rv = model_instance.getScalar('Rv','Wall','Rv','VWall')
        Ym = np.array(model_instance.getVector('Sv','TriSeg','Sv','Y'))
        
        signCm_Sv = np.sign(model_instance.getVector('Sv','Wall','Sv','Cm'))
        signCm_Rv = np.sign(model_instance.getVector('Rv','Wall','Rv','Cm'))
        
        # for convergence reason, do not reject unhealthy simulations
        if np.any(Am_Sv/np.pi - Ym**2 < 0):
            return 100/self.constRVD
        
        Xm_Sv = signCm_Sv * np.sqrt(Am_Sv/np.pi - Ym**2) + VWall_Sv/Am_Sv / 2
        Xm_Rv = signCm_Rv * np.sqrt(Am_Rv/np.pi - Ym**2) - VWall_Rv/Am_Rv/ 2
        
        RVD = np.max(Xm_Rv-Xm_Sv)*1e3
        
        self.scalarData = {'LVEF':LVEFmod, 'EDV': np.max(Vmod)*1e6, 'RVD': RVD}

    def setModelConstants(self,model_instance):
        if not(self.tCycle==[]):
            model_instance.setScalar('','','','tCycle',self.tCycle)

        model_instance.setScalar('','','','q0',self.getQ0())

    def setModelConstantsSteps(self,model_instance,stepSize):
        if not(self.tCycle==[]):
            tCycle = model_instance.getScalar('','','','tCycle')
            model_instance.setScalar('','','','tCycle',tCycle * (1-stepSize) + self.tCycle * stepSize)
            model_instance.runFast()

        q0 = model_instance.getScalar('','','','q0')
        model_instance.setScalar('','','','q0',q0 * (1-stepSize) + stepSize * self.getQ0())
        model_instance.runFast()

    def getComputedData(self,keyToGive,func,*args, **kwargs):
        if keyToGive in self.computedData:
            return self.computedData[keyToGive]
        # compute data
        res = func(*args,**kwargs)
        self.computedData[keyToGive] = res
        return res

    def getQ0(self,returnDefault=True):
        if 'LVEDV_Echo' in self.scalarData and 'LVEF_Echo' in self.scalarData:
            tCycle = self.getTCycle()
            EF = self.scalarDAta['LVEF_Echo']
            EDV= self.scalarDAta['LVEDV_Echo']

            SV = EDV * EF / 100 * 1e-6
            q0 = EDV * tCycle
            return q0

        if 'RVSV_MRI' in self.scalarData:
            return self.scalarData['RVSV_MRI']*1e-6 / self.tCycle

        if returnDefault:
            return 8.5000e-05
        return float('nan')

    def getTCycle(self,returnDefault=True):
        return self.tCycle
