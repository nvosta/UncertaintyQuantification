######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# Handles Parameters to the model
######

import numpy as np
import copy

sigmoid = lambda x: 1 / (1+np.exp(-x))
sigmoid_inv = lambda x : np.log( x / (1 - x) )

class ParametersManual20:
    '''
        Handles Parameters
        Set values in the model
    '''
    
    VWallConstLv = 0.009791426396109512;
    VWallConstSv = 0.006624461812148345;
    VWallConstRv = 0.004854809874169944;
    
    LDADADOratio = 1.0570 / 0.65
    
    locnames = ['LVfw','IVS', 'RV apex', 'RV mid', 'RV base','LVfw','IVS', 'RV apex', 'RV mid', 'RV base','LVfw','IVS', 'RV apex', 'RV mid', 'RV base','LVfw','IVS', 'RV', 'global']
    
    def __init__(self,parameters=[],coupledParametersRelative=[],coupledParametersAbsolute=[]):
        self.parameters = []
        

    def __len__(self):
        return 20

    def getX(self,circadapt):
        '''
            Get current parameter state, normalized if needed
        '''
        X = self.getParameterValues(circadapt)
        X = self.parValtoX(X)
        return X

    def getParameterValues(self,circadapt):
        '''
            Get current parameter state, actual values
        '''
        # Get Parameters
        X = np.zeros((20))
        X[0] = circadapt.getScalar('Lv','Patch','Lv1','SfAct')
        X[1] = circadapt.getScalar('Sv','Patch','Sv1','SfAct')
        X[2] = circadapt.getScalar('Rv','Patch','Rv1','SfAct')
        X[3] = circadapt.getScalar('Rv','Patch','Rv2','SfAct')
        X[4] = circadapt.getScalar('Rv','Patch','Rv3','SfAct')
        X[5] = circadapt.getScalar('Lv','Patch','Lv1','k1')
        X[6] = circadapt.getScalar('Sv','Patch','Sv1','k1')
        X[7] = circadapt.getScalar('Rv','Patch','Rv1','k1')
        X[8] = circadapt.getScalar('Rv','Patch','Rv2','k1')
        X[9] = circadapt.getScalar('Rv','Patch','Rv3','k1')
        X[10] = circadapt.getScalar('Lv','Patch','Lv1','dT')
        X[11] = circadapt.getScalar('Sv','Patch','Sv1','dT')
        X[12] = circadapt.getScalar('Rv','Patch','Rv1','dT')
        X[13] = circadapt.getScalar('Rv','Patch','Rv2','dT')
        X[14] = circadapt.getScalar('Rv','Patch','Rv3','dT')
        X[15] = circadapt.getScalar('Lv','Patch','Lv1','AmRef')
        X[16] = circadapt.getScalar('Sv','Patch','Sv1','AmRef')
        X[17] = circadapt.getScalar('Rv','Patch','Rv1','AmRef')
        X[18] = circadapt.getScalar('Rv','Patch','Rv1','ADO')
        X[19] = circadapt.getScalar('','','','q0')
        return X
    
    def setParameterValues(self,circadapt, X):
        '''
            Get current parameter state, actual values
        '''
        circadapt.setScalar('Lv','Patch','Lv1','SfAct', X[0])
        circadapt.setScalar('Sv','Patch','Sv1','SfAct', X[1])
        circadapt.setScalar('Rv','Patch','Rv1','SfAct', X[2])
        circadapt.setScalar('Rv','Patch','Rv2','SfAct', X[3])
        circadapt.setScalar('Rv','Patch','Rv3','SfAct', X[4])
        circadapt.setScalar('Lv','Patch','Lv1','k1', X[5])
        circadapt.setScalar('Sv','Patch','Sv1','k1', X[6])
        circadapt.setScalar('Rv','Patch','Rv1','k1', X[7])
        circadapt.setScalar('Rv','Patch','Rv2','k1', X[8])
        circadapt.setScalar('Rv','Patch','Rv3','k1', X[9])
        circadapt.setScalar('Lv','Patch','Lv1','dT', X[10])
        circadapt.setScalar('Sv','Patch','Sv1','dT', X[11])
        circadapt.setScalar('Rv','Patch','Rv1','dT', X[12])
        circadapt.setScalar('Rv','Patch','Rv2','dT', X[13])
        circadapt.setScalar('Rv','Patch','Rv3','dT', X[14])
        circadapt.setScalar('Lv','Patch','Lv1','AmRef', X[15])
        circadapt.setScalar('Sv','Patch','Sv1','AmRef', X[16])
        circadapt.setScalar('Rv','Patch','Rv1','AmRef', X[17])
        circadapt.setScalar('Rv','Patch','Rv2','AmRef', X[17])
        circadapt.setScalar('Rv','Patch','Rv3','AmRef', X[17])

        circadapt.setScalar('Lv','Patch','Lv1','VWall', X[15]*self.VWallConstLv)
        circadapt.setScalar('Sv','Patch','Sv1','VWall', X[16]*self.VWallConstSv)
        circadapt.setScalar('Rv','Patch','Rv1','VWall', X[17]*self.VWallConstRv)
        circadapt.setScalar('Rv','Patch','Rv2','VWall', X[17]*self.VWallConstRv)
        circadapt.setScalar('Rv','Patch','Rv3','VWall', X[17]*self.VWallConstRv)

        circadapt.setScalar('Lv','Patch','Lv1','ADO', X[18])
        circadapt.setScalar('Sv','Patch','Sv1','ADO', X[18])
        circadapt.setScalar('Rv','Patch','Rv1','ADO', X[18])
        circadapt.setScalar('Rv','Patch','Rv2','ADO', X[18])
        circadapt.setScalar('Rv','Patch','Rv3','ADO', X[18])
        circadapt.setScalar('Lv','Patch','Lv1','LDAD', X[18]*self.LDADADOratio)
        circadapt.setScalar('Sv','Patch','Sv1','LDAD', X[18]*self.LDADADOratio)
        circadapt.setScalar('Rv','Patch','Rv1','LDAD', X[18]*self.LDADADOratio)
        circadapt.setScalar('Rv','Patch','Rv2','LDAD', X[18]*self.LDADADOratio)
        circadapt.setScalar('Rv','Patch','Rv3','LDAD', X[18]*self.LDADADOratio)

        circadapt.setScalar('','','','q0', X[19])
        
        return X

    def getParameterValue(self,iPar,circadapt=[],xNorm =[]):
        '''
            Get current parameter state, actual values
        '''
        # Get Parameters
        return 0

    def setX(self,circadapt,X):
        '''
            Put numpy array of X in circadapt
        '''
        X = self.XtoParVal(np.array(X))
        self.setParameterValues(circadapt, X)

    def setXsingle(self,circadapt,p0,p1,p2,p3,x,iPar, lb=[],ub=[]):
        pass

    def outOfBound(self,X):
        return out_of_bound(X)
    
    def out_of_bound(self,X,iPar=-1):
        # else
        par = self.XtoParVal(np.array(X))
        if np.any(par[:5]<0):
            return True
        if np.any(par[5:10]<1):
            return True
        
        return False         

    def getNames(self,r=[]):
        return [self.getName(i) for i in range(20)]

    def getName(self,i):
        return ''

    def getNumberOfParameters(self):
        return 20

    def parameterGetScale(self,parName=[],iPar=[]):
        if iPar < 5:
            return 1e3
        if iPar < 10:
            return 1
        if iPar < 15:
            return 1e-3
        if iPar < 18:
            return 1e-4
        if iPar == 19:
            return 1/60000
        return 1

    def parameterGetLabelModel(self,parName=[],iPar=[],includeLocation=False, isTheta=False):
        loc = ''
        if includeLocation:
            if iPar in [0, 5, 10, 15]:
                loc = ' LV'
            if iPar in [1, 6, 11, 16]:
                loc = ' IVS'
            if iPar in [2, 7, 12]:
                loc = ' RV apex'
            if iPar in [3, 8, 13]:
                loc = ' RV mid'
            if iPar in [4, 9, 14]:
                loc = ' RV base'
            if iPar == 17:
                loc = ' RV'
        if iPar < 5:
            return 'SfAct' + loc
        if iPar < 10:
            return 'k1' + loc
        if iPar < 15:
            return 'dT' + loc
        if iPar < 18:
            return 'AmRef' + loc
        if iPar < 19:
            return 'ADO+LDAD'
        if iPar < 20:
            return 'CO'
        return ''

    def parameterGetLabelClinical(self,parName=[],iPar=[], isTheta=False):
        if iPar<5:
            return 'Contractility'
        if iPar<10:
            return 'Stiffness'
        if iPar<15:
            return 'Activation delay'
        if iPar < 18:
            return 'Wall area'
        if iPar < 19:
            return 'TimeFac'
        if iPar < 20:
            return 'Cardiac Output'
        return ''

    def parameterGetUnit(self,parName=[],iPar=[], isTheta=False):
        if iPar < 5:
            return 'kPa'
        if iPar < 10:
            return '-'
        if iPar < 15:
            return 'ms'
        if iPar < 18 or iPar == 20:
            return 'cm^2'
        if iPar == 19:
            return 'L/min'
        return '-'

    def XtoParVal(self,X,iPar=-1):
        if iPar>-1:
            x1 = []
            x2 = np.zeros(20)
            
            if len(np.array(X).shape)==0:
                x2[iPar] = X
                return self.XtoParVal(x2)[iPar]
            
            for iSim in range(len(X)):
                x2[iPar] = X[iSim]
                x1.append(self.XtoParVal(x2)[iPar])
            return np.array(x1)
        
        if len(np.array(X).shape)==2:
            x = []
            for iSim in range(len(X)):
                x.append(self.XtoParVal(X[iSim]))
            return np.array(x)
                
        X = self.XtoParValSingleSim(X)
        if iPar>-1:
            return X[:,iPar]
        return X

    def parValtoX(self,X,iPar=-1):
        if iPar>-1:
            X = np.ones(20)*X
        X = self.parValtoXSingleSim(X)
        if iPar>-1:
            return X[:,iPar]
        return X

    def XtoParValSingleSim(self,X):
        parval = np.ndarray(self.getNumberOfParameters())
        
        # old
        # SfAct
        #parval[:10] = 2**X[:10]
        #dT
        #parval[10:15] = X[10:15]
        # rest
        #parval[15:] = 2**X[15:]
        
        # SfAct
        parval[0:5] = sigmoid(X[0:5]) * 1e6
        
        # k1
        parval[5:10] = sigmoid(X[5:10]) * 99 + 1
        
        # dT
        parval[10:15] = sigmoid(X[10:15]) - 0.2
        
        # rest is >0
        parval[15:] = 2**X[15:]
        
        
        
        return parval

    def parValtoXSingleSim(self,parval):        
        # SfAct, log betweeon -20 and 20 -> ~0 - 1e6
        #X[:10] = np.log2(X[:10])
        #X[10:15] = X[10:15]
        #X[15:] = np.log2(X[15:])
        
        X = np.ndarray(len(self))
        # SfAct
        X[0:5] = sigmoid_inv(parval[0:5]/1e6)
        # k1
        X[5:10] = sigmoid_inv((parval[5:10]-1)/99)
        # dT
        X[10:15] = sigmoid_inv((parval[10:15]+0.2))
        # rest is >0
        X[15:] = np.log2(parval[15:])
        
        return X
    
    def get_log_correction(self, X):
        parval = self.XtoParVal(np.array(X))
        logcorrection = 0
        # SfAct
        logcorrection = logcorrection - np.sum(np.log( 
            sigmoid(X[:,0:5] ) * (1-sigmoid(X[:,0:5] )) / 1e6
            ), axis=1)
        logcorrection = logcorrection - np.sum(np.log( 
            sigmoid(X[:,5:10] ) * (1-sigmoid(X[:,5:10] )) / 99
            ), axis=1)
        logcorrection = logcorrection - np.sum(np.log( 
            sigmoid(X[:,10:15] ) * (1-sigmoid(X[:,10:15] )) / 1
            ), axis=1)
        
        # rest
        logcorrection = logcorrection - np.sum(np.log(parval[:,15:]), axis=1)
        
        return logcorrection
    
class ParametersManual20_NOVWALL(ParametersManual20):
    def setParameterValues(self,circadapt, X):
        '''
            Get current parameter state, actual values
        '''
        circadapt.setScalar('Lv','Patch','Lv1','SfAct', X[0])
        circadapt.setScalar('Sv','Patch','Sv1','SfAct', X[1])
        circadapt.setScalar('Rv','Patch','Rv1','SfAct', X[2])
        circadapt.setScalar('Rv','Patch','Rv2','SfAct', X[3])
        circadapt.setScalar('Rv','Patch','Rv3','SfAct', X[4])
        circadapt.setScalar('Lv','Patch','Lv1','k1', X[5])
        circadapt.setScalar('Sv','Patch','Sv1','k1', X[6])
        circadapt.setScalar('Rv','Patch','Rv1','k1', X[7])
        circadapt.setScalar('Rv','Patch','Rv2','k1', X[8])
        circadapt.setScalar('Rv','Patch','Rv3','k1', X[9])
        circadapt.setScalar('Lv','Patch','Lv1','dT', X[10])
        circadapt.setScalar('Sv','Patch','Sv1','dT', X[11])
        circadapt.setScalar('Rv','Patch','Rv1','dT', X[12])
        circadapt.setScalar('Rv','Patch','Rv2','dT', X[13])
        circadapt.setScalar('Rv','Patch','Rv3','dT', X[14])
        circadapt.setScalar('Lv','Patch','Lv1','AmRef', X[15])
        circadapt.setScalar('Sv','Patch','Sv1','AmRef', X[16])
        circadapt.setScalar('Rv','Patch','Rv1','AmRef', X[17])
        circadapt.setScalar('Rv','Patch','Rv2','AmRef', X[17])
        circadapt.setScalar('Rv','Patch','Rv3','AmRef', X[17])

        #circadapt.setScalar('Lv','Patch','Lv1','VWall', X[15]*self.VWallConstLv)
        #circadapt.setScalar('Sv','Patch','Sv1','VWall', X[16]*self.VWallConstSv)
        #circadapt.setScalar('Rv','Patch','Rv1','VWall', X[17]*self.VWallConstRv)
        #circadapt.setScalar('Rv','Patch','Rv2','VWall', X[17]*self.VWallConstRv)
        #circadapt.setScalar('Rv','Patch','Rv3','VWall', X[17]*self.VWallConstRv)

        circadapt.setScalar('Lv','Patch','Lv1','ADO', X[18])
        circadapt.setScalar('Sv','Patch','Sv1','ADO', X[18])
        circadapt.setScalar('Rv','Patch','Rv1','ADO', X[18])
        circadapt.setScalar('Rv','Patch','Rv2','ADO', X[18])
        circadapt.setScalar('Rv','Patch','Rv3','ADO', X[18])
        circadapt.setScalar('Lv','Patch','Lv1','LDAD', X[18]*self.LDADADOratio)
        circadapt.setScalar('Sv','Patch','Sv1','LDAD', X[18]*self.LDADADOratio)
        circadapt.setScalar('Rv','Patch','Rv1','LDAD', X[18]*self.LDADADOratio)
        circadapt.setScalar('Rv','Patch','Rv2','LDAD', X[18]*self.LDADADOratio)
        circadapt.setScalar('Rv','Patch','Rv3','LDAD', X[18]*self.LDADADOratio)

        circadapt.setScalar('','','','q0', X[19])
        
        return X
    
class ParametersManual20_NOVWALL_smallSfAct(ParametersManual20_NOVWALL):
    def XtoParValSingleSim(self,X):
        parval = np.ndarray(self.getNumberOfParameters())
        
        # old
        # SfAct
        #parval[:10] = 2**X[:10]
        #dT
        #parval[10:15] = X[10:15]
        # rest
        #parval[15:] = 2**X[15:]
        
        # SfAct
        parval[0:5] = sigmoid(X[0:5]) * 2.5e5
        
        # k1
        parval[5:10] = sigmoid(X[5:10]) * 99 + 1
        
        # dT
        parval[10:15] = sigmoid(X[10:15]) - 0.2
        
        # rest is >0
        parval[15:] = 2**X[15:]
        
        
        
        return parval

    def parValtoXSingleSim(self,parval):        
        # SfAct, log betweeon -20 and 20 -> ~0 - 1e6
        #X[:10] = np.log2(X[:10])
        #X[10:15] = X[10:15]
        #X[15:] = np.log2(X[15:])
        
        X = np.ndarray(len(self))
        # SfAct
        X[0:5] = sigmoid_inv(parval[0:5]/2.5e5)
        # k1
        X[5:10] = sigmoid_inv((parval[5:10]-1)/99)
        # dT
        X[10:15] = sigmoid_inv((parval[10:15]+0.2))
        # rest is >0
        X[15:] = np.log2(parval[15:])
        
        return X
    
    def get_log_correction(self, X):
        parval = self.XtoParVal(np.array(X))
        logcorrection = 0
        # SfAct
        logcorrection = logcorrection - np.sum(np.log( 
            sigmoid(X[:,0:5] ) * (1-sigmoid(X[:,0:5] )) / 2.5e5
            ), axis=1)
        logcorrection = logcorrection - np.sum(np.log( 
            sigmoid(X[:,5:10] ) * (1-sigmoid(X[:,5:10] )) / 99
            ), axis=1)
        logcorrection = logcorrection - np.sum(np.log( 
            sigmoid(X[:,10:15] ) * (1-sigmoid(X[:,10:15] )) / 1
            ), axis=1)
        
        # rest
        logcorrection = logcorrection - np.sum(np.log(parval[:,15:]), axis=1)
        
        return logcorrection