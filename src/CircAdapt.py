#######
# Python class for CircAdapt
# handles all CircAdapt functions
######

from ctypes import windll, cdll, c_double,  c_char_p,c_char, c_bool, POINTER, c_int
import ctypes;
import scipy.io as sio
import numpy as np

######
# Functions needed for loadMat
######
def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

class CircAdapt:
    '''
        CircAdapt: Handles the c++ code
    '''
    # Init CircAdapt by telling where to find the dll file
    def __init__(self,pathToCircAdapt,filename='',Pdict=[]):
        self.pathToCircAdapt = pathToCircAdapt
        self.CA = []
        self.handle = []
        if not(filename==''):
            self.buildTriSeg()
            if filename[-4:]=='.mat':
                self.loadMat(filename)
            else:
                self.loadDict(filename)
        if not(Pdict==[]):
            self.buildTriSeg()
            self.setPdict(Pdict)

    def __del__(self):
        self.destroyCA()

    def destroyCA(self):
        if not(self.CA==[]):
            self.CA.deleteCore()

        self.CA = []
    ######
    # Build functions
    ######

    # build TriSeg: Walmsley 2015
    def buildTriSeg(self):
        self.initCA()
        self.CA.buildTriSeg()

    ######
    # General initalization
    ######
    def initCA(self):
        self.destroyCA()
        self.CA = cdll.LoadLibrary(self.pathToCircAdapt)

        # getScalar
        self.CA.getScalar.restype = c_double
        self.CA.getScalar.argtypes = [c_char_p,c_char_p,c_char_p,c_char_p]

        # setScalar
        self.CA.setScalar.restype = c_bool
        self.CA.setScalar.argtypes = [c_char_p,c_char_p,c_char_p,c_char_p,c_double]

        #get Vector Length
        self.CA.getVecLength.restype = c_int

        # Get actual length of vector
        self.CA.getVector.argtypes = [c_char_p,c_char_p,c_char_p,c_char_p,POINTER(c_double)]

        #get Vector Length
        self.CA.getVersion.restype = c_char_p

        self.CA.getIsStable.restype = c_bool
    ######
    # Functions to run beats in CircAdapt
    ######

    # single beat
    def runSingleBeat(self):
        self.CA.runSingleBeat();

    # single beat
    def runStable(self):
        self.CA.runStable(True);

    ######
    # Communicate data
    ######

    # get a scalar variable
    def getScalar(self,Loc1,Mod,Loc2,Par):
        Loc1 = Loc1.encode('ASCII');
        Mod = Mod.encode('ASCII');
        Loc2 = Loc2.encode('ASCII');
        Par = Par.encode('ASCII');
        return self.CA.getScalar(Loc1,Mod,Loc2,Par)

    def getIsStable(self):
        return self.CA.getIsStable()

    # get vector data
    def getVector(self, v1, v2=[], v3=[], v4=[]):
        # Get vector getVecLength
        vecLen = self.CA.getVecLength()

        # init empty data array
        data = (c_double * vecLen)()  # equivalent to C++ double[vecLen]

        if v2==[] and v3==[] and v4==[]:
            par = v1.encode('ASCII');

            # Get actual length of vector
            self.CA.getVector1(par, data)
        elif v4==[]:
            loc = v1.encode('ASCII');
            mod = v2.encode('ASCII');
            par = v3.encode('ASCII');

            # Get actual length of vector
            self.CA.getVector3(loc, mod, par, data)

        else:
            loc0 = v1.encode('ASCII');
            mod = v2.encode('ASCII');
            loc1 = v3.encode('ASCII');
            par = v4.encode('ASCII');

            # Get actual length of vector
            self.CA.getVector(loc0, mod, loc1, par, data)

        return list(data)

    def getAllTimeSteps(self):
        loc1 = ''
        loc2 = 'TimeAll'

        loc1 = loc1.encode('ASCII');
        loc2 = loc2.encode('ASCII');

        vecLen = self.CA.getAllVecLength()
        data = (c_double * vecLen)()  # equivalent to C++ double[vecLen]
        self.CA.getVector(loc1,loc1,loc1,loc2,data)
        return list(data)

    # update variables in CircAdapt
    def setScalar(self,Loc1,Mod,Loc2,Par,val):
        Loc1 = Loc1.encode('ASCII');
        Mod = Mod.encode('ASCII');
        Loc2 = Loc2.encode('ASCII');
        Par = Par.encode('ASCII');
        return self.CA.setScalar(Loc1,Mod,Loc2,Par,c_double(val))

    ######
    # LOAD and SAVE files
    ######

    # TODO:
    # load, load file optimized for Python
    # save, save file optimzed for Python

    # laodMat
    def loadMat(self,filename,structname='P'):
        P = _check_keys(sio.loadmat(filename, struct_as_record=False, squeeze_me=True))[structname]
        self.setPdict(P)
        return P

    def setPdict(self,P):
        # in future, check which version we should build here
        self.CA.buildTriSeg()

        # send data to CA object
        self.setScalar('','','','rhob',P['General']['rhob'])
        self.setScalar('','','','q0',P['General']['q0'])
        self.setScalar('','','','p0',P['General']['p0'])
        self.setScalar('','','','tCycle',P['General']['tCycle'])
        self.setScalar('','','','FacpControl',P['General']['FacpControl'])

        # ScaleVqY
        self.setScalar('','','','dt',P['General']['Dt'])
        self.setScalar('','','','tCycleRest',P['General']['tCycleRest'])
        self.setScalar('','','','TimeFac',P['General']['TimeFac'])
        self.setScalar('','','','PressFlowContr',P['General']['PressFlowContr'])
        self.setScalar('','','','dTauAv',P['General']['dTauAv'])

        # ArtVen
        par = {'k','Len','A0','p0','AWall'};
        for p in par:
            self.setScalar('CiSy','ArtVen','Art',p,P['ArtVen'][p][0,0]);
            self.setScalar('CiSy','ArtVen','Ven',p,P['ArtVen'][p][1,0]);
            self.setScalar('CiPu','ArtVen','Art',p,P['ArtVen'][p][0,1]);
            self.setScalar('CiPu','ArtVen','Ven',p,P['ArtVen'][p][1,1]);

        par = {'p0AV','q0AV','kAV'};
        for p in par:
            self.setScalar('CiSy','ArtVen','',p,P['ArtVen'][p][0]);
            self.setScalar('CiPu','ArtVen','',p,P['ArtVen'][p][1]);

        # Chamber
        # no values have to go from Chamber

        # TriSeg
        self.setScalar('Lv','TriSeg','','tau',P['TriSeg']['Tau']);
        self.setScalar('Lv','TriSeg','','V',P['TriSeg']['V'][-1]);
        self.setScalar('Lv','TriSeg','','Y',P['TriSeg']['Y'][-1]);

        # iCavity
        self.setScalar('CiSy','Cavity','Art','V',P['Cavity']['V'][-1][0])
        self.setScalar('CiSy','Cavity','Ven','V',P['Cavity']['V'][-1][1])
        self.setScalar('CiPu','Cavity','Art','V',P['Cavity']['V'][-1][2])
        self.setScalar('CiPu','Cavity','Ven','V',P['Cavity']['V'][-1][3])
        self.setScalar('La','Cavity','La','V',P['Cavity']['V'][-1][4])
        self.setScalar('Ra','Cavity','Ra','V',P['Cavity']['V'][-1][5])
        self.setScalar('Lv','Cavity','Lv','V',P['Cavity']['V'][-1][6])
        self.setScalar('Rv','Cavity','Rv','V',P['Cavity']['V'][-1][7])

        # Wall
        loc1 = {'La','Ra','Lv','Sv','Rv'}
        for loc in loc1:
            self.setScalar(loc,'Wall','','AmDead',P['Wall']['AmDead'][P['Wall']['Name']==loc]);
            if P['Wall']['nPatch'][P['Wall']['Name']==loc]>1:
                self.setScalar(loc,'Wall',loc + '1','Split',P['Wall']['nPatch'][P['Wall']['Name']==loc] )

        # patch
        par = {'dT','LsRef','Ls0Pas','dLsPas','SfPas','k1','Lsi0Act','LenSeriesElement','SfAct','ADO','vMax','TimeAct','TR','TD','CRest','VWall','AmRef'};
        for iL in range(len(P['Patch']['Name'])):
            for p in par:
                if p in P['Patch'].keys():
                    v = P['Patch'][p][iL]
                    self.setScalar(P['Patch']['Name'][iL][0:2],'Patch',P['Patch']['Name'][iL],p,v)
                else:
                    print('key ' + p + 'not found')
        par = {'Lsi','C'};
        for iL in range(len(P['Patch']['Name'])):
            for p in par:
                if p in P['Patch'].keys():
                    v = P['Patch'][p][-1][iL]
                    self.setScalar(P['Patch']['Name'][iL][0:2],'Patch',P['Patch']['Name'][iL],p,v)
                else:
                    print('key ' + p + 'not found')

        # Bag
        self.setScalar('Peri','Bag','','k',P['Bag']['k'])
        self.setScalar('Peri','Bag','','VRef',P['Bag']['VRef'])
        self.setScalar('Peri','Bag','','SfPeri',P['Bag']['pAdapt'])

        # Valves
        par = {'AOpen','ALeak','Len'}
        for iL in range(len(P['Valve']['Name'])):
            for p in par:
                if p in P['Valve'].keys():
                    v = P['Valve'][p][iL]
                    self.setScalar(P['Valve']['Name'][iL],'Valve',P['Valve']['Name'][iL],p,v)
        par = {'q'}
        for iL in range(len(P['Valve']['Name'])):
            for p in par:
                if p in P['Valve'].keys():
                    v = P['Valve'][p][-1][iL]
                    self.setScalar(P['Valve']['Name'][iL],'Valve',P['Valve']['Name'][iL],p,v)

    def getPdict(self):
        '''
            Save CircAdapt to .mat structure
        '''
        # init P dictionary
        P = {}
        P['Log']={}
        P['General']={}
        P['ArtVen']={}
        P['Chamber']={}
        P['TriSeg']={}
        P['Cavity']={}
        P['Wall']={}
        P['Patch']={}
        P['Node']={}
        P['Bag']={}
        P['Valve']={}
        P['Adapt']={}

        # get Time
        P['t'] = self.getVector('','','','Time')
        t = np.ndarray((len(P['t']),1))
        t[:,0] = self.getVector('','','','Time')
        P['t']=t

        # Log
        P['Log']['Code']='cpp'
        P['Log']['Version']=self.CA.getVersion()
        P['Log']['WrapperLanguage']='Python'

        # General
        P['General']['rhob']           = self.getScalar('','','','rhob');
        P['General']['q0']             = self.getScalar('','','','q0');
        P['General']['p0']             = self.getScalar('','','','p0');
        P['General']['tCycle']         = self.getScalar('','','','tCycle');
        P['General']['FacpControl']    = self.getScalar('','','','FacpControl');
        P['General']['ScaleVqY']       = [1e-05,0.0001,0.1];
        P['General']['Dt']             = self.getScalar('','','','dt');
        P['General']['tCycleRest']     = self.getScalar('','','','tCycleRest');
        P['General']['TimeFac']        = self.getScalar('','','','TimeFac');
        P['General']['PressFlowContr'] = self.getScalar('','','','PressFlowContr');
        P['General']['TauAv']          = self.getScalar('','','','TauAv');
        P['General']['dTauAv']         = self.getScalar('','','','dTauAv');

        # ArtVen
        artvenname= np.zeros((2,), dtype=np.object)
        artvenname[0] = 'Sy'
        artvenname[1] = 'Pu'
        P['ArtVen']['Name'] = artvenname;
        P['ArtVen']['n'] = 2;
        P['ArtVen']['iCavity'] = [1, 3];
        P['ArtVen']['iWall'] = [1, 3];
        P['ArtVen']['Adapt'] = {};

        par = {'k','Len','A0','p0','AWall'}
        for p in par:
            P['ArtVen'][p] = np.ndarray((2,2))
            P['ArtVen'][p][0,0] = self.getScalar('CiSy','ArtVen','Art',p)
            P['ArtVen'][p][1,0] = self.getScalar('CiSy','ArtVen','Ven',p)
            P['ArtVen'][p][0,1] = self.getScalar('CiPu','ArtVen','Art',p)
            P['ArtVen'][p][1,1] = self.getScalar('CiPu','ArtVen','Ven',p)
        par = {'p0AV','q0AV','kAV'};
        for p in par:
            P['ArtVen'][p] = np.ndarray(2)
            P['ArtVen'][p][0] = self.getScalar('CiSy','ArtVen','',p);
            P['ArtVen'][p][1] = self.getScalar('CiPu','ArtVen','',p);
        par = {'q'};
        for p in par:
            P['ArtVen'][p] = np.ndarray((len(P['t']),2))
            P['ArtVen'][p][:,0] = self.getVector('CiSy','ArtVen','',p);
            P['ArtVen'][p][:,1] = self.getVector('CiPu','ArtVen','',p);

        # Chamber
        chambername= np.zeros((2,), dtype=np.object)
        chambername[0] = 'La'
        chambername[1] = 'Ra'
        P['Chamber']['Name'] = chambername;
        P['Chamber']['n'] = 2;
        P['Chamber']['iCavity'] = [5, 6];
        P['Chamber']['iWall'] = [5, 6];

        # TriSeg
        trisegname= np.zeros((1,), dtype=np.object)
        trisegname[0] = 'v'
        P['TriSeg']['Name'] = trisegname;
        P['TriSeg']['n'] = 1;
        P['TriSeg']['iCavity'] = 7;
        P['TriSeg']['iWall'] = 7;

        par = {'V','Y','VS','YS','VDot','YDot'};
        for p in par:
            P['TriSeg'][p] = np.ndarray((len(P['t']),1))
            P['TriSeg'][p][:,0] = self.getVector('','TriSeg','',p);
        P['TriSeg']['Tau'] = self.getScalar('Lv','TriSeg','','tau');


        # Cavity
        cavityname= np.zeros((8,), dtype=np.object)
        cavityname[0] = 'SyArt'
        cavityname[1] = 'SyVen'
        cavityname[2] = 'PuArt'
        cavityname[3] = 'PuVen'
        cavityname[4] = 'La'
        cavityname[5] = 'Ra'
        cavityname[6] = 'Lv'
        cavityname[7] = 'Rv'
        P['Cavity']['Name'] = cavityname;
        P['Cavity']['n'] = 8;
        P['Cavity']['iNode'] = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
        P['Cavity']['Adapt'] = {};
        par = ['V','A','Z','p','VDot'];
        cpppar = ['V','A','Z','pCavity','VDot'];
        Loc1 = ['CiSy','CiSy','CiPu','CiPu','La','Ra','Lv','Rv'];
        Loc2 = ['Art','Ven','Art','Ven','','','',''];
        for iP in range(len(par)):
            P['Cavity'][par[iP]] = np.ndarray((len(P['t']),8))
            for iL in range(len(Loc1)):
                P['Cavity'][par[iP]][:,iL] = self.getVector(Loc1[iL],'Cavity',Loc2[iL],cpppar[iP]);

        # Wall
        wallname= np.zeros((10,), dtype=np.object)
        wallname[0] = 'SyArt'
        wallname[1] = 'SyVen'
        wallname[2] = 'PuArt'
        wallname[3] = 'PuVen'
        wallname[4] = 'La'
        wallname[5] = 'Ra'
        wallname[6] = 'Lv'
        wallname[7] = 'Sv'
        wallname[8] = 'Rv'
        wallname[9] = 'Peri'
        P['Wall']['Name'] = wallname;
        P['Wall']['n'] = 10;
        P['Wall']['nPatch'] = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]);
        P['Wall']['iPatch'] = np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6]);
        # Ventricular Patch to Wall
        par = ['AmDead','VWall'];
        Loc1 = ['La','Ra','Lv','Sv','Rv'];
        for iP in range(len(par)):
            P['Wall'][par[iP]] = np.ndarray(10)
            P['Wall'][par[iP]][:]=0
            for iL in range(len(Loc1)):
                P['Wall'][par[iP]][iL+4] = self.getScalar(Loc1[iL],'Wall','',par[iP])
        par = ['Am0','DADT','T','Cm','Am','Vm','pTrans'];
        Loc1 = ['La','Ra','Lv','Sv','Rv'];
        for iP in range(len(par)):
            P['Wall'][par[iP]] = np.ndarray((len(P['t']),10))
            P['Wall'][par[iP]][:,:]=0
            for iL in range(len(Loc1)):
                P['Wall'][par[iP]][:,iL+4] = self.getVector(Loc1[iL],'Wall','',par[iP])

        # get patch data
        allWalls=['La','Ra','Lv','Sv','Rv']
        nPatch = 0
        for iWall in range(len(allWalls)):
            P['Wall']['iPatch'][iWall+4] = int(P['Wall']['iPatch'][iWall+3] + nPatch)
            nPatch = self.getScalar(allWalls[iWall],'Wall','','nPatch')
            P['Wall']['nPatch'][iWall+4] = int(nPatch)


        # patch
        nPatch = int(sum(P['Wall']['nPatch']))
        patchnames = np.zeros((nPatch,), dtype=np.object)
        iPatchName = 0
        for iWall in range(len(wallname)):
            for iPatch in range(P['Wall']['nPatch'][iWall]):
                patchnames[P['Wall']['iPatch'][iWall]-1+iPatch]=wallname[iWall] + str(iPatch+1)

        P['Patch']['Name'] = patchnames
        P['Patch']['n'] = nPatch

        par = ['dT','ActivationDelay','LsRef','Ls0Pas','dLsPas','SfPas','k1','Lsi0Act','LenSeriesElement','SfAct','ADO','vMax','TimeAct','TR','TD','CRest','VWall','AmRef'];
        for p in par:
            P['Patch'][p] = np.ndarray(nPatch)
            P['Patch'][p][:]=0
            for iP in range(len(patchnames)):
                P['Patch'][p][iP] = self.getScalar(patchnames[iP][0:2],'Patch',patchnames[iP],p);
        par = ['Lsi','C','T','Ef','Ls','CDot','SfEcm','SfPasT','LsiDot','Sf','DSfDEf','DADT','Am0','Am'];
        for p in par:
            P['Patch'][p] = np.ndarray((len(P['t']),nPatch))
            P['Patch'][p][:]=0
            for iP in range(len(patchnames)):
                P['Patch'][p][:,iP] = self.getVector(patchnames[iP][0:2],'Patch',patchnames[iP],p);
        P['Patch']['Adapt'] = {}

        # Node
        P['Node']['Name'] = cavityname;
        P['Node']['n'] = 8;
        P['Node']['iCavity'] = [1, 2, 3, 4, 5, 6, 7, 8];
        par = ['q','Y','p'];
        Loc1 = ['CiSy','CiSy','CiPu','CiPu','La','Ra','Lv','Rv'];
        Loc2 = ['Art','Ven','Art','Ven','','','',''];
        Mod = ['ArtVen','ArtVen','ArtVen','ArtVen','Chamber','Chamber','Cavity','Cavity'];
        for iP in range(len(par)):
            P['Node'][par[iP]] = np.ndarray((len(P['t']),8))
            for iL in range(len(Loc1)):
                P['Node'][par[iP]][:,iL] = self.getVector(Loc1[iL],Mod[iL],Loc2[iL],par[iP]);

        # Bag
        bagnames= np.zeros((1,), dtype=np.object)
        bagnames[0] = 'Peri'
        P['Bag']['Name'] = cavityname;
        P['Bag']['n'] = 1;
        P['Bag']['iWall'] = [10];
        P['Cavity']['Adapt'] = {};
        P['Bag']['pAdapt'] = self.getScalar('Peri','Bag','','SfPeri');
        P['Bag']['p'] = self.getVector('Peri','Bag','','p');
        P['Bag']['k'] = self.getScalar('Peri','Bag','','k');
        P['Bag']['VRef'] = self.getScalar('Peri','Bag','','VRef');

        # Valve
        valvenames= np.zeros((9,), dtype=np.object)
        valvenames[0] = 'SyVenRa'
        valvenames[1] = 'RaRv'
        valvenames[2] = 'RvPuArt'
        valvenames[3] = 'PuVenLa'
        valvenames[4] = 'LaLv'
        valvenames[5] = 'LvSyArt'
        valvenames[6] = 'LaRa'
        valvenames[7] = 'LvRv'
        valvenames[8] = 'SyArtPuArt'
        P['Valve']['Name'] = valvenames;
        P['Valve']['n'] = 9;
        P['Valve']['iNodeProx'] = [2, 6, 8, 4, 5, 7, 5, 7, 1];
        P['Valve']['iNodeDist'] = [6, 8, 3, 5, 7, 1, 6, 8, 3];

        par = ['AOpen','ALeak','Len'];
        for iP in range(len(par)):
            P['Valve'][par[iP]] = np.ndarray(9)
            for iL in range(len(valvenames)):
                P['Valve'][par[iP]][iL] = self.getScalar(valvenames[iL],'Valve','',par[iP]);
        par = ['q','L','A','qDot'];
        for iP in range(len(par)):
            P['Valve'][par[iP]] = np.ndarray((len(P['t']),9))
            for iL in range(len(valvenames)):
                P['Valve'][par[iP]][:,iL] = self.getVector(valvenames[iL],'Valve','',par[iP]);
        return P

    def getPdictCompressed(self):
        P = self.getPdict()
        P = self.compressP(P,len(P['t']))
        return P

    def compressP(self,P,lenVec):
        for k, v in P.items():
            if type(v) == dict:
                P[k] = self.compressP(v,lenVec)
            elif type(v) == list or type(v) == np.ndarray:
                if len(v)==lenVec:
                    P[k] = np.array([v[-1]])
        return P

    def saveMat(self,filename,structname='P'):
        P = self.getPdict()
        # Save P dictionary to matlab struct
        sio.savemat(filename,{structname:P})

    def saveDict(self,filename):
        P = self.getPdict()
        np.save(filename,P,allow_pickle=True)

    def loadDict(self,filename):
        P = np.load(filename,allow_pickle=True).item()
        self.setPdict(P)

    def isSuccess(self):
        '''
            Return false if vectors contain nan
        '''
        return not(np.any(np.isnan(self.getVector('LvSyArt','Valve','','q')))) and self.getIsStable()
