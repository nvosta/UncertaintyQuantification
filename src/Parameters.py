######
# CircAdapt Sensitivity Analysis and Parameter Estimation
# Nick van Osta
# Handles Parameters to the model
######

import numpy as np
import copy

class Parameters:
    '''
        Handles Parameters
        Set values in the model
    '''
    def __init__(self,parameters=[],coupledParametersRelative=[],coupledParametersAbsolute=[]):
        self.parameters = []
        self.isNormalized = []
        self.isNormalizedList = []
        self.lb = []
        self.ub = []
        self.isLog10 = []
        self.isLog2 = []
        self.isLog = []
        self.loglb = []
        self.logub = []
        self.addParameters(parameters)
        self.coupledParametersRelative=coupledParametersRelative
        self.coupledParametersAbsolute=coupledParametersAbsolute

    def __len__(self):
        return len(self.parameters)

    def addParameters(self,parameters):
        '''
            For initialization, add parameters
        '''
        for p in parameters:
            self.parameters.append(p[0:4])
            isNormalized = False
            isNormalizedList = False
            lb = 0
            ub = 0
            isLog10 = False
            isLog = False
            isLog2 = False
            loglb = np.nan
            logub = np.nan
            if len(p)>4:
                iP = 4
                while iP<len(p):
                    if p[iP]=='lb':
                        iP=iP+1
                        lb=p[iP]
                        if type(lb)==list:
                            isNormalizedList=True
                        else:
                            isNormalized=True
                    elif p[iP]=='ub':
                        iP=iP+1
                        ub=p[iP]
                        if type(lb)==list:
                            isNormalizedList=True
                        else:
                            isNormalized=True
                    elif p[iP]=='log':
                        isLog = True

                        iP=iP+1
                        loglb=p[iP]
                        iP=iP+1
                        logub=p[iP]
                    elif p[iP]=='log10':
                        isLog10 = True
                    elif p[iP]=='log2':
                        isLog2 = True
                    else:
                        raise Exception('Unknown keyword')
                    iP=iP+1
                    
            
            self.isNormalized.append(isNormalized)
            self.isNormalizedList.append(isNormalizedList)
            self.lb.append(lb)
            self.ub.append(ub)
            self.isLog10.append(isLog10)
            self.isLog2.append(isLog2)
            self.isLog.append(isLog)
            self.loglb.append(loglb)
            self.logub.append(logub)
            
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)

    def getX(self,circadapt):
        '''
            Get current parameter state, normalized if needed
        '''
        # Get parameter values
        X = self.getParameterValues(circadapt)
        
        
        # log10
        for iP in range(len(X)):
            if self.isLog10[iP]:
                X[iP] = np.math.log10(X[iP])
            elif self.isLog2[iP]:
                if self.parameters[iP][3]=='k1':
                    X[iP] = np.math.log2(X[iP]-1)
                else:
                    X[iP] = np.math.log2(X[iP])

        # Normalize
        lb = np.array(self.lb)
        ub = np.array(self.ub)
        X[self.isNormalized] = (X[self.isNormalized] - lb[self.isNormalized]) /  (ub[self.isNormalized] - lb[self.isNormalized])

        for iP in range(len(X)):
            if self.isNormalizedList[iP]:
                X[iP] = (X[iP] - lb[iP][0]) /  (ub[iP][0] - lb[iP][0])
            if self.isLog[iP]:
                print('X: ', X[iP])
                print('loglb: ', self.loglb[iP])
                print('logub: ', self.logub[iP])
                X[iP] = np.log(
                    self.loglb[iP] + (self.logub[iP] - self.loglb[iP])*X[iP])
                
                # normalize
                print('potlb', np.log(self.loglb[iP]) )
                print('potub', np.log(self.logub[iP]) )
                X[iP] = (X[iP] - np.log(self.loglb[iP])) / (
                    np.log(self.logub[iP]) -  np.log(self.loglb[iP]))


        # Coupled values Relative
        for cPR in self.coupledParametersRelative:
            XcPR = np.array(X[cPR])

            XcPR[0] = np.mean(X[cPR])
            XcPR[1:] = XcPR[1:] / XcPR[0]

            X[cPR] = XcPR[:]

        for cPR in self.coupledParametersRelative:
            XcPR = np.array(X[cPR])

            XcPR[1:] = XcPR[1:] - XcPR[0]

            X[cPR] = XcPR[:]
            
        

        # Return
        return X

    def getParameterValues(self,circadapt):
        '''
            Get current parameter state, actual values
        '''
        # Get Parameters
        X = np.ndarray(len(self.parameters))
        for iP in range(len(self.parameters)):
            p=self.parameters[iP]
            if type(p[2])==list:
                X[iP] = circadapt.getScalar(p[0],p[1],p[2][0],p[3])
            else:
                X[iP] = circadapt.getScalar(p[0],p[1],p[2],p[3])
        return X

    def getParameterValue(self,iPar,circadapt=[],xNorm =[]):
        '''
            Get current parameter state, actual values
        '''
        # Get Parameters
        X = np.ndarray(len(self.parameters))
        if not(circadapt==[]):
            p=self.parameters[iPar]
            if type(p[2])==list:
                return circadapt.getScalar(p[0],p[1],p[2][0],p[3])
            else:
                return circadapt.getScalar(p[0],p[1],p[2],p[3])
        else:
            lb = np.array(self.lb)[iPar]
            ub = np.array(self.ub)[iPar]
            return xNorm * (ub-lb)+lb


    def printValues(self,model_instance):
        values = self.getParameterValues(model_instance)
        for iP in range(len(self.parameters)):
            print(self.parameters[iP][0],self.parameters[iP][1],self.parameters[iP][2],self.parameters[iP][3],values[iP])


    def setX(self,circadapt,X):
        '''
            Put numpy array of X in circadapt
        '''
        # Denormalize
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        # Coupled values Relative
        for cPR in self.coupledParametersRelative:
            XcPR = np.array(X[cPR])

            XcPR[1:] = XcPR[1:] * XcPR[0]
            XcPR[0] =  ( len(cPR) * np.mean(XcPR[0]) ) - np.sum(XcPR[1:])

            X[cPR] = XcPR[:]

        for cPR in self.coupledParametersRelative:
            XcPR = np.array(X[cPR])

            XcPR[1:] = XcPR[1:] + XcPR[0]

            X[cPR] = XcPR[:]
            
        # Log
        for iP in range(len(X)):
            if self.isLog[iP]:
                
                X[iP] = X[iP] * np.log(self.logub[iP]) -  np.log(self.loglb[iP]) + np.log(self.loglb[iP])
                X[iP] = (np.exp(X[iP]) - self.loglb[iP]) / (self.logub[iP] - self.loglb[iP])

        # Put variable
        for iP in range(len(self.parameters)):
            p=self.parameters[iP]

            self.setXsingle(circadapt,p[0],p[1],p[2],p[3],X[iP], iP,self.lb[iP],self.ub[iP])

    def setXsingle(self,circadapt,p0,p1,p2,p3,x,iPar, lb=[],ub=[]):
        if type(p2)==list:
            for iP in range(len(p2)):
                if type(lb)==list and len(lb)>0:
                    self.setXsingle(circadapt,p0,p1,p2[iP],p3,x,iPar,lb[iP],ub[iP])
                else:
                    self.setXsingle(circadapt,p0,p1,p2[iP],p3,x,iPar, lb,ub)
        else:
            if not(lb==[]):
                x = x * (ub-lb)+lb
            # log10
            if self.isLog10[iPar]:
                x = 10**x
            elif self.isLog2[iPar]:
                if self.parameters[iPar][3]=='k1':
                    x = 2**x+1
                else:
                    x = 2**x
            circadapt.setScalar(p0,p1,p2,p3,x)

    def outOfBound(self,X):
        for iP in range(len(X)):
            par = self.XtoParVal(X[iP], iP)
            
            # natural boundaries, list not complete, add parameters if needed
            parMinZero = ['SfAct','SfPas','VWall','AmRef','tCycle','q0','TimeFac']
            parMinOne  = ['k1']
            if self.parameters[iP][3] in parMinZero and par < 0:
                if not self.isLog10[iP] and not self.isLog2[iP] and not self.isLog[iP]:
                    return True
            if self.parameters[iP][3] in parMinOne and par < 1:
                if not self.isLog10[iP] and not self.isLog2[iP] and not self.isLog[iP]:
                    return True
        return False
    
    def out_of_bound(self,X,iPar=-1):
        if iPar<0:
            for iPar in range(len(X)):
                if self.out_of_bound(X[iPar],iPar):
                    return True
            return False
        # else
        par = self.XtoParVal(X, iPar)
        
        # natural boundaries, list not complete, add parameters if needed
        parMinZero = ['SfAct','SfPas','VWall','AmRef','tCycle','q0','TimeFac']
        parMinOne  = ['k1']
        if self.parameters[iPar][3] in parMinZero and par < 0:
            return True
        if self.parameters[iPar][3] in parMinOne and par < 1:
            return True
        return False       

    def getNames(self,r=[]):
        names = []
        if r == []:
            r = range(len(self.parameters))
        for i in r:
            names.append(self.getName(i))
        return names

    def getName(self,i):
        p=self.parameters[i]
        if type(p[2])==list:
            p2=p[0]
        else:
            p2 = p[2]
        return p[3]+' '+p2

    def getNumberOfParameters(self):
        return len(self.parameters)

    def parameterGetScale(self,parName=[],iPar=[]):
        if parName==[]:
            parName = self.parameters[iPar][3]
        d = {'SfAct':1e3,'dT':1e-3,'q0':5e-5/3,'AmRef':1e-4,'dTauAv':1e-3}
        if parName in d:
            return d[parName]
        return 1

    def parameterGetLabelModel(self,parName=[],iPar=[],includeLocation=False):
        if parName==[]:
            parName = self.parameters[iPar][3]
        d = {}
        if parName in d:
            parName = d[parName]
        if includeLocation:
            loc = self.parameters[iPar][2]
            if type(loc)==list:
                loc = self.parameters[iPar][0]
            parName = parName + ' ' + loc
        return parName

    def parameterGetLabelClinical(self,parName=[],iPar=[]):
        if parName==[]:
            parName = self.parameters[iPar][3]
        d = {'SfAct':'Contractility','dT':'Activation Delay','q0':'Cardiac Output','k1':'Stiffness'}
        if parName in d:
            return d[parName]
        return self.parameterGetLabelModel(parName=parName,iPar=iPar)

    def parameterGetUnit(self,parName=[],iPar=[]):
        if parName==[]:
            parName = self.parameters[iPar][3]
        d = {'SfAct':'kPa','dT':'ms','q0':'L/min','AmRef':r'$cm^2$','dTauAv':'ms'}
        if parName in d:
            return d[parName]
        return '-'

    def XtoParVal(self,X,iPar=-1):
        if iPar==-1:
            #X=X*(self.ub-self.lb) + self.lb
            X1 = [self.XtoParVal(X[:,i], i) for i in range(X.shape[1])]
            X = np.array(X1).transpose()
        else:
            if self.isLog[iPar]:
                X = X * np.log(self.logub[iPar]) -  np.log(self.loglb[iPar]) + np.log(self.loglb[iPar])
                X = (np.exp(X) - self.loglb[iPar]) / (self.logub[iPar] - self.loglb[iPar])
            X=X*(self.ub[iPar]-self.lb[iPar]) + self.lb[iPar]

            if self.isLog10[iPar]:
                X = 10**X
            elif self.isLog2[iPar]:
                if self.parameters[iPar][3]=='k1':
                    X = 2**X+1
                else:
                    X = 2**X
        return X

    def parValtoX(self,X,iPar=-1):
        if iPar==-1:
            for iP in range(len(X)):
                if self.isLog10[iP]:
                    X[iP] = np.math.log10(X[iP])
                elif self.isLog2[iP]:
                    X[iP] = np.math.log2(X[iP])
            X= ( X - self.lb ) / (self.ub-self.lb)
        else:
            if self.isLog10[iP]:
                X[iP] = np.math.log10(X[iP])
            elif self.isLog2[iP]:
                X[iP] = np.math.log2(X[iP])
            X= ( X - self.lb[iPar] ) / (self.ub[iPar]-self.lb[iPar])
          
        return X


