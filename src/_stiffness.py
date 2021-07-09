import numpy as np
import matplotlib.pyplot as plt

def plotStiffness(model_instance,idx0=''):

    model_instance.run()

    # Plot stiffness based on tissue properties
    Pdict = model_instance.getPdict()
    parLsRef  = np.array(Pdict['Patch']['LsRef'][4:])
    pardLsPas = np.array(Pdict['Patch']['dLsPas'][4:])
    parLs0Pas = np.array(Pdict['Patch']['Ls0Pas'][4:])
    parSfPas = np.array(Pdict['Patch']['SfPas'][4:])
    parK1 = np.array(Pdict['Patch']['k1'][4:])
    parSfAct = np.array(Pdict['Patch']['SfAct'][4:])
    parVWALL = np.array(Pdict['Patch']['VWall'][4:])
    parAmRef = np.array(Pdict['Patch']['AmRef'][4:])

    idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))
    idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))

    if idx0=='onsetQRS':
        idxED = model_instance.onsetQRS()
    elif idx0=='PVcorner':
        V = Pdict['Node']['p'][:,6]
        p = Pdict['Cavity']['V'][:,6]

        #plt.plot(p,V)
        #plt.plot(p[:-2],np.diff(V,2)*1e3)

        idxED = np.argmax(np.diff(V,2)[:100])
        #plt.scatter(p[idxED],V[idxED])
        #plt.show()

    Ls = Pdict['Patch']['Ls'][idxED,4:]
    kk3 = 2 * parLsRef / pardLsPas
    LfP = Ls / parLs0Pas
    y = LfP**parK1
    yTit = LfP**kk3
    DSfPasDEf = y * (0.0349*parSfPas*parK1) + yTit * (0.01*parSfAct*kk3);
    SfEcm = (y - 1)*(0.0349*parSfPas);
    SfPasT = SfEcm+(yTit - 1)*(0.01*parSfAct);
    ls0  = np.log(Ls/parLs0Pas) - SfPasT/DSfPasDEf

    plt.scatter(np.log(Ls/parLs0Pas),SfPasT,ec=[0,0,0],fc=[1,1,1],zorder=999,label='Formula based')

    for iLoc in range(3):
        plt.plot([ls0[iLoc],ls0[iLoc]+np.log(Ls[iLoc]/parLs0Pas[iLoc])],[0,np.log(Ls[iLoc]/parLs0Pas[iLoc])*DSfPasDEf[iLoc]],c=[0,0,0],zorder=1,linewidth=2,label='Formula based' if iLoc==0 else None)
        plt.text(ls0[iLoc]+np.log(Ls[iLoc]/parLs0Pas[iLoc]), np.log(Ls[iLoc]/parLs0Pas[iLoc])*DSfPasDEf[iLoc], 'dy/dx = ' + str(int(np.round(DSfPasDEf[iLoc]))) )


    plotLs = np.linspace(ls0-0.1,ls0+0.15,100)
    plotSf = []
    for iP in range(len(plotLs)):
        LfP = np.exp(plotLs[iP])
        y = LfP**parK1
        yTit = LfP**kk3
        SfEcm = (y - 1)*(0.0349*parSfPas);
        SfPasT = SfEcm+(yTit - 1)*(0.01*parSfAct);
        plotSf.append(SfPasT)
    plt.plot(plotLs,plotSf,c=[0,0,0],zorder=998,linewidth=1 )


    #### Q0 based
    if False:
        q0 = model_instance.getScalar('','','','q0')
        q0Space = np.linspace(0.9*q0,1.1*q0,5)
    
        EfQ0 = []
        SfQ0 = []
        for iQ in range(len(q0Space)):
            print('iQ')
            model_instance.setScalar('','','','q0',q0Space[iQ])
            model_instance.run()
    
            Pdict = model_instance.getPdict()
    
            idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))
            idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))
            if idx0=='onsetQRS':
                idxED = model_instance.onsetQRS()
            elif idx0=='PVcorner':
                V = Pdict['Node']['p'][:,6]
                p = Pdict['Cavity']['V'][:,6]
                idxED = np.argmax(np.diff(V,2)[:100])
    
            Ls = Pdict['Patch']['Ls'][idxED,4:]
            Sf = Pdict['Patch']['Sf'][idxED,4:]
    
            print(' - idxED: ', idxED)
            print(' - Ls: ', Ls)
            print(' - Sf: ', Sf)
    
            plt.scatter(np.log(Ls/parLs0Pas),Sf,10,fc=[1,0,0],zorder=9,label='Q0 based' if iQ==0 else None)
    
            EfQ0.append(np.log(Ls/parLs0Pas))
            SfQ0.append(Sf)
    
        EfQ0=np.array(EfQ0)
        SfQ0 = np.array(SfQ0)
    
        for iLoc in range(3):
            z = np.polyfit(EfQ0[:,iLoc], SfQ0[:,iLoc], 1)
            p = np.poly1d(z)
            print(z)
            xp = np.linspace(ls0[iLoc], ls0[iLoc]+0.2, 100)
            plt.plot(xp, p(xp), '-',c=[1,0,0],label='Q0 based' if iLoc==0 else None)
            plt.text(xp[-1],p(xp[-1]), 'dy/dx = ' + str(int(np.round(z[0]))) , c=[1,0,0] )

    plt.legend()

    plt.xlabel('Ef [-]')
    plt.ylabel('Sf [Pa]')


def plotCompliance(model_instance,idx0='',cost=[],patient=[]):

    model_instance.run()

    # Plot stiffness based on tissue properties
    Pdict = model_instance.getPdict()
    parLsRef  = np.array(Pdict['Patch']['LsRef'][4:])
    pardLsPas = np.array(Pdict['Patch']['dLsPas'][4:])
    parLs0Pas = np.array(Pdict['Patch']['Ls0Pas'][4:])
    parSfPas = np.array(Pdict['Patch']['SfPas'][4:])
    parK1 = np.array(Pdict['Patch']['k1'][4:])
    parSfAct = np.array(Pdict['Patch']['SfAct'][4:])
    parVWALL = np.array(Pdict['Patch']['VWall'][4:])
    parAmRef = np.array(Pdict['Patch']['AmRef'][4:])

    idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))
    idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))

    if not cost==[]:
        print(cost)
        idxED = cost.getIDX0(patient=patient,model_instance=model_instance)

    if idx0=='onsetQRS':
        idxED = model_instance.onsetQRS()
    elif idx0=='PVcorner':
        V = Pdict['Node']['p'][:,6]
        p = Pdict['Cavity']['V'][:,6]

        #plt.plot(p,V)
        #plt.plot(p[:-2],np.diff(V,2)*1e3)

        idxED = np.argmax(np.diff(V,2)[:100])
        #plt.scatter(p[idxED],V[idxED])
        #plt.show()

    if not cost==[]:
        print(cost)
        idxED = cost.getIDX0(patient=patient,model_instance=model_instance)

    Ls = Pdict['Patch']['Ls'][idxED,4:]
    kk3 = 2 * parLsRef / pardLsPas
    LfP = Ls / parLs0Pas
    y = LfP**parK1
    yTit = LfP**kk3
    DSfPasDEf = y * (0.0349*parSfPas*parK1) + yTit * (0.01*parSfAct*kk3);
    SfEcm = (y - 1)*(0.0349*parSfPas);
    SfPasT = SfEcm+(yTit - 1)*(0.01*parSfAct);
    ls0  = np.log(Ls/parLs0Pas) - SfPasT/DSfPasDEf

    plt.scatter(np.array(SfPasT)/1e3,(np.exp(np.log(Ls/parLs0Pas)-np.log(Ls/parLs0Pas))-1)*100,ec=[0,0,0],fc=[1,1,1],zorder=999,label='Formula based')

    for iLoc in range(3):
        #plt.plot([0,np.log(Ls[iLoc]/parLs0Pas[iLoc])*DSfPasDEf[iLoc]], [ls0[iLoc],ls0[iLoc]+np.log(Ls[iLoc]/parLs0Pas[iLoc])],c=[0,0,0],zorder=1,linewidth=2,label='Formula based' if iLoc==0 else None)
        #plt.plot([0,np.log(Ls[iLoc]/parLs0Pas[iLoc])*DSfPasDEf[iLoc]], [np.exp(ls0[iLoc]-np.log(Ls/parLs0Pas)),np.exp(ls0[iLoc]+np.log(Ls[iLoc]/parLs0Pas[iLoc]) -np.log(Ls/parLs0Pas))],c=[0,0,0],zorder=1,linewidth=2,label='Formula based' if iLoc==0 else None)
        plt.text(ls0[iLoc]+np.log(Ls[iLoc]/parLs0Pas[iLoc]), np.log(Ls[iLoc]/parLs0Pas[iLoc])*DSfPasDEf[iLoc], 'dy/dx = ' + str(int(np.round(DSfPasDEf[iLoc]))) )


    plotLs = np.linspace(ls0-0.1,ls0+0.15,100)
    plotSf = []
    for iP in range(len(plotLs)):
        LfP = np.exp(plotLs[iP])
        y = LfP**parK1
        yTit = LfP**kk3
        SfEcm = (y - 1)*(0.0349*parSfPas);
        SfPasT = SfEcm+(yTit - 1)*(0.01*parSfAct);
        plotSf.append(SfPasT)
    plt.plot(np.array(plotSf)/1e3,(np.exp(plotLs-np.log(Ls/parLs0Pas))-1)*100,c=[0,0,0],zorder=998,linewidth=1 )


    #### Q0 based
    q0 = model_instance.getScalar('','','','q0')
    q0Space = np.linspace(0.95*q0,1.05*q0,5)

    Ef0 = np.log(Ls/parLs0Pas)

    EfQ0 = []
    SfQ0 = []
    for iQ in range(len(q0Space)):
        print('iQ')
        model_instance.setScalar('','','','q0',q0Space[iQ])
        model_instance.run()

        Pdict = model_instance.getPdict()

        idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))
        idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))
        if idx0=='onsetQRS':
            idxED = model_instance.onsetQRS()
        elif idx0=='PVcorner':
            V = Pdict['Node']['p'][:,6]
            p = Pdict['Cavity']['V'][:,6]
            idxED = np.argmax(np.diff(V,2)[:100])

        Ls = Pdict['Patch']['Ls'][idxED,4:]
        Sf = Pdict['Patch']['Sf'][idxED,4:]

        print(' - idxED: ', idxED)
        print(' - Ls: ', Ls)
        print(' - Sf: ', Sf)

        plt.scatter(Sf/1e3,(np.exp(np.log(Ls/parLs0Pas)-Ef0)-1)*100,10,fc=[1,0,0],zorder=9,label='Q0 based' if iQ==0 else None)

        EfQ0.append((np.exp(np.log(Ls/parLs0Pas)-Ef0)-1)*100)
        SfQ0.append(Sf/1e3)

    EfQ0=np.array(EfQ0)
    SfQ0 = np.array(SfQ0)

    for iLoc in range(3):
        z = np.polyfit(SfQ0[:,iLoc],EfQ0[:,iLoc], 1)
        p = np.poly1d(z)
        print(z)
        xp = np.linspace(0.5*np.min(SfQ0[:,iLoc]),np.max(SfQ0[:,iLoc])*2, 2)
        plt.plot(xp, p(xp), '-',c=[1,0,0],label='Q0 based' if iLoc==0 else None)
        plt.text(xp[-1],p(xp[-1]), 'dy/dx = ' + str(np.round(z[0]*1e3)) + '%/MPa' , c=[1,0,0] )

    plt.legend()

    plt.ylabel('Strain [%]')
    plt.xlabel(r'$\Delta$Sf [kPa]')


def plotContractility(model_instance):
    model_instance.run()

    Pdict = model_instance.getPdict()

    Ls = Pdict['Patch']['Ls'][:,4:]
    Sf = Pdict['Patch']['Sf'][:,4:]*1e-3
    dT = Pdict['Patch']['dT'][4:] + Pdict['Patch']['ActivationDelay'][4:]
    SfPasT = Pdict['Patch']['SfPasT'][:,4:]*1e-3

    t = Pdict['t']*1e3

    for iP in range(3):
        plt.plot(t,Sf[:,iP]-SfPasT[:,iP], linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('SfActT')

        maxSfAct=np.max(Sf[:,iP]-SfPasT[:,iP])
        argmaxSfAct=np.argmax(Sf[:,iP]-SfPasT[:,iP])

        idxStart = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] > 0.25*maxSfAct)[0,0]
        idxEnd = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] < 0.75*maxSfAct   )[-1,-1]
        plt.scatter(t[idxStart:idxEnd],Sf[idxStart:idxEnd,iP]-SfPasT[idxStart:idxEnd,iP], linewidth=3)

        z = np.polyfit(t[idxStart:idxEnd].transpose()[0], Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP], 1)
        p = np.poly1d(z)
        t0 = t[idxStart] - (Sf[idxStart,iP] - SfPasT[idxStart,iP]) / z[0]
        xp = np.linspace(t0, t[idxEnd], 3)
        plt.plot(xp, p(xp), '-',c=[1,0,0],label='Contractility' if iP==0 else None)
        plt.text(xp[-1],p(xp[-1]), 'dy/dx = ' + str(int(np.round(z[0]))) , c=[1,0,0] )


        # Activation Time
        argmaxSfActDiff=np.argmax(np.diff(Sf[:idxEnd,iP]-SfPasT[:idxEnd,iP],2))
        plt.scatter(t[argmaxSfActDiff],Sf[argmaxSfActDiff,iP]-SfPasT[argmaxSfActDiff,iP])

        idxStart = argmaxSfActDiff-5
        idxEnd = argmaxSfActDiff+5
        z = np.polyfit(t[idxStart:idxEnd].transpose()[0], Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP], 1)
        p = np.poly1d(z)

        t0 = t[idxStart] - (Sf[idxStart,iP] - SfPasT[idxStart,iP]) / z[0]
        xp = np.linspace(t0, t[idxEnd], 3)
        #plt.plot(xp, p(xp), '-',c=[1,0,0],label='Activation' if iP==0 else None)
        #plt.plot([t0,t0],[0,-500])


        #plt.plot([dT[iP],dT[iP]],[0,-1000])
    plt.xticks([0,400,800])

def plotDADTpass(model_instance):
    model_instance.run()

    Pdict = model_instance.getPdict()
    
    idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))
    
    
    Pdict = model_instance.getPdict()
    
    Ls = Pdict['Patch']['Ls'][:,4:]
    Am = Pdict['Patch']['Am'][:,4:]
    C = Pdict['Patch']['C'][:,4:]
    Am0 = Pdict['Patch']['Am0'][:,4:]
    T = Pdict['Patch']['T'][:,4:]
    parLsRef  = np.array(Pdict['Patch']['LsRef'][4:])
    pardLsPas = np.array(Pdict['Patch']['dLsPas'][4:])
    parLs0Pas = np.array(Pdict['Patch']['Ls0Pas'][4:])
    parSfPas = np.array(Pdict['Patch']['SfPas'][4:])
    parK1 = np.array(Pdict['Patch']['k1'][4:])
    parSfAct = np.array(Pdict['Patch']['SfAct'][4:])
    parVWALL = np.array(Pdict['Patch']['VWall'][4:])
    parAmRef = np.array(Pdict['Patch']['AmRef'][4:])

    kk3 = 2 * parLsRef / pardLsPas
    LfP = Ls / parLs0Pas
    y = LfP**parK1
    yTit = LfP**kk3
    DSfPasDEf = y * (0.0349*parSfPas*parK1) + yTit * (0.01*parSfAct*kk3);
    SfEcm = (y - 1)*(0.0349*parSfPas);
    SfPasT = SfEcm+(yTit - 1)*(0.01*parSfAct);
    ls0  = np.log(Ls/parLs0Pas) - SfPasT/DSfPasDEf

    DADT = Am**2 / (DSfPasDEf - 2 * SfPasT) / (0.25*parVWALL)
    
    dC = np.diff(C, axis=0)
    print(dC.shape)
    dC = np.concatenate((dC, dC[-1].reshape(1,-1)), axis=0)
    
    idx=np.any(C>0.001*np.max(C), axis=1)
    
    #plt.plot(T, Am0+DADT*T,'--')
    T[idx]=np.nan
    plt.plot(T, (Am0+DADT*T)*1e4)
    plt.scatter(T[idxED,:], (Am0+DADT*T)[idxED,:]*1e4, fc=[1,1,1], ec=[0,0,0], zorder=10)
    
    Trange = T[idxED,:] + np.linspace(-2.5, 2.5, 2).reshape(-1,1)
    
    plt.plot(Trange, (Am0[idxED,:]+DADT[idxED,:]*Trange)*1e4)
    
def plotWork(model_instance):
    Pdict = model_instance.getPdict()
    Ef = Pdict['Patch']['Ef'][:,4:]
    Sf = Pdict['Patch']['Sf'][:,4:]*1e-3
    
    plt.plot(Ef, Sf)
    

def plotAfterloadRV(model_instance,cost=[],patient=[]):
    model_instance.run()

    Pdict = model_instance.getPdict()

    Ls = Pdict['Patch']['Ls'][:,4:]
    Sf = Pdict['Patch']['Sf'][:,4:]
    dT = Pdict['Patch']['dT'][4:] + Pdict['Patch']['ActivationDelay'][4:]
    SfPasT = Pdict['Patch']['SfPasT'][:,4:]

    idxED = cost.getIDX0(patient=patient,model_instance=model_instance)

    idxES = np.argmin(Pdict['Cavity']['V'][:,6])

    Strain = (Ls / Ls[idxED,:] - 1 ) * 100


    for iP in range(3):


        plt.plot(Strain[:,iP],Sf[:,iP])

        plt.scatter(Strain[idxED,iP],Sf[idxED,iP],fc=[.5,.5,.5])
        plt.scatter(Strain[idxES,iP],Sf[idxES,iP],fc=[.5,.5,.5])

        plt.plot([Strain[idxED,iP],Strain[idxES,iP]],[Sf[idxED,iP],Sf[idxES,iP]],c=[.5,.5,.5])






def getContractilityRV(Pdict=[],model_instance=[]):
    if Pdict==[]:
        Pdict = model_instance.getPdict()

    Ls = Pdict['Patch']['Ls'][:,4:]
    Sf = Pdict['Patch']['Sf'][:,4:]
    dT = Pdict['Patch']['dT'][4:] + Pdict['Patch']['ActivationDelay'][4:]
    SfPasT = Pdict['Patch']['SfPasT'][:,4:]

    t = Pdict['t']

    contraRV = []

    for iP in range(3):

        maxSfAct=np.max(Sf[:,iP]-SfPasT[:,iP])
        argmaxSfAct=np.argmax(Sf[:,iP]-SfPasT[:,iP])

        idxStart = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] > 0.25*maxSfAct)[0,0]
        idxEnd = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] < 0.75*maxSfAct   )[-1,-1]

        z = np.polyfit(t[idxStart:idxEnd].transpose()[0], Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP], 1)
        p = np.poly1d(z)

        contraRV.append(z[0])

    return contraRV

def getContractilityTensionRV(Pdict=[],model_instance=[]):
    if Pdict==[]:
        Pdict = model_instance.getPdict()

    Ls = Pdict['Patch']['Ls'][:,4:]
    Sf = Pdict['Patch']['Sf'][:,4:]
    dT = Pdict['Patch']['dT'][4:] + Pdict['Patch']['ActivationDelay'][4:]
    SfPasT = Pdict['Patch']['SfPasT'][:,4:]
    Am = Pdict['Patch']['Am'][:,4:]

    t = Pdict['t']

    contraRV = []

    for iP in range(3):

        maxSfAct=np.max(  (Sf[:,iP]-SfPasT[:,iP]) * Am[:,iP] )
        argmaxSfAct=np.argmax(  (Sf[:,iP]-SfPasT[:,iP] ) * Am[:,iP])

        idxStart = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] > 0.25*maxSfAct)[0,0]
        idxEnd = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] < 0.75*maxSfAct   )[-1,-1]

        z = np.polyfit(t[idxStart:idxEnd].transpose()[0], (Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP])* Am[idxStart:idxEnd,iP], 1)
        p = np.poly1d(z)

        contraRV.append(z[0])

    return contraRV



def getDelayRV(Pdict=[],model_instance=[]):
    if Pdict==[]:
        Pdict = model_instance.getPdict()

    Ls = Pdict['Patch']['Ls'][:,4:]
    Sf = Pdict['Patch']['Sf'][:,4:]
    dT = Pdict['Patch']['dT'][4:] + Pdict['Patch']['ActivationDelay'][4:]
    SfPasT = Pdict['Patch']['SfPasT'][:,4:]

    t = Pdict['t']

    delayRV = []

    for iP in range(3):
        maxSfAct=np.max(Sf[:,iP]-SfPasT[:,iP])
        argmaxSfAct=np.argmax(Sf[:,iP]-SfPasT[:,iP])

        idxStart = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] > 0.25*maxSfAct)[0,0]
        idxEnd = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] < 0.75*maxSfAct   )[-1,-1]
        z = np.polyfit(t[idxStart:idxEnd].transpose()[0], Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP], 1)
        p = np.poly1d(z)

        t0 = t[idxStart] - (Sf[idxStart,iP] - SfPasT[idxStart,iP]) / z[0]
        delayRV.append(t0[0])

    return delayRV

def getComplianceRV(Pdict=[],model_instance=[],cost=[],patient=[]):
    if Pdict==[]:
        Pdict = model_instance.getPdict()

    parLsRef  = np.array(Pdict['Patch']['LsRef'][4:])
    pardLsPas = np.array(Pdict['Patch']['dLsPas'][4:])
    parLs0Pas = np.array(Pdict['Patch']['Ls0Pas'][4:])
    parSfPas = np.array(Pdict['Patch']['SfPas'][4:])
    parK1 = np.array(Pdict['Patch']['k1'][4:])
    parSfAct = np.array(Pdict['Patch']['SfAct'][4:])
    parVWALL = np.array(Pdict['Patch']['VWall'][4:])
    parAmRef = np.array(Pdict['Patch']['AmRef'][4:])


    idxED = cost.getIDX0(patient=patient,model_instance=model_instance)

    Ls = Pdict['Patch']['Ls'][idxED,4:]
    kk3 = 2 * parLsRef / pardLsPas
    LfP = Ls / parLs0Pas
    y = LfP**parK1
    yTit = LfP**kk3
    DSfPasDEf = y * (0.0349*parSfPas*parK1) + yTit * (0.01*parSfAct*kk3);
    SfEcm = (y - 1)*(0.0349*parSfPas);
    SfPasT = SfEcm+(yTit - 1)*(0.01*parSfAct);
    ls0  = np.log(Ls/parLs0Pas) - SfPasT/DSfPasDEf





    #### Q0 based
    q0 = model_instance.getScalar('','','','q0')
    q0Space = np.linspace(0.95*q0,1.05*q0,5)

    allPdict = []
    allPdict.append(Pdict)

    Ef0 = np.log(Ls/parLs0Pas)

    EfQ0 = []
    SfQ0 = []
    for iQ in range(len(q0Space)):
        print('iQ')
        model_instance.setScalar('','','','q0',q0Space[iQ])
        model_instance.run()

        iTest = 0
        while not model_instance.getIsStable() and iTest<len(allPdict):
            model_instance.setPdict(allPdict[iTest])
            model_instance.setScalar('','','','q0',q0Space[iQ])
            model_instance.run()
            iTest = iTest+1

        Pdict = model_instance.getPdict()
        allPdict.append(Pdict)


        idxED = cost.getIDX0(patient=patient,model_instance=model_instance)

        Ls = Pdict['Patch']['Ls'][idxED,4:]
        Sf = Pdict['Patch']['Sf'][idxED,4:]

        print(' - idxED: ', idxED)
        print(' - Ls: ', Ls)
        print(' - Sf: ', Sf)


        EfQ0.append((np.exp(np.log(Ls/parLs0Pas)-Ef0)-1)*100)
        SfQ0.append(Sf/1e3)

    EfQ0=np.array(EfQ0)
    SfQ0 = np.array(SfQ0)

    complianceRV=[]

    for iLoc in range(3):
        z = np.polyfit(SfQ0[:,iLoc],EfQ0[:,iLoc], 1)
        p = np.poly1d(z)
        complianceRV.append(z[0])
        xp = np.linspace(0.5*np.min(SfQ0[:,iLoc]),np.max(SfQ0[:,iLoc])*2, 2)

    return complianceRV


def get(model_instance,what,loc, idx0='onsetQRS'):
    Pdict = model_instance.getPdict()
    
    Am = Pdict['Patch']['Am'][:,loc]
    VWall = Pdict['Patch']['VWall'][loc]
    Ls = Pdict['Patch']['Ls'][:,loc]
    Sf = Pdict['Patch']['Sf'][:,loc]
    Ef = Pdict['Patch']['Ef'][:,loc]
    dT = Pdict['Patch']['dT'][loc] + Pdict['Patch']['ActivationDelay'][loc]
    SfPasT = Pdict['Patch']['SfPasT'][:,loc]
    
    parLsRef  = np.array(Pdict['Patch']['LsRef'][loc])
    pardLsPas = np.array(Pdict['Patch']['dLsPas'][loc])
    parLs0Pas = np.array(Pdict['Patch']['Ls0Pas'][loc])
    parSfPas = np.array(Pdict['Patch']['SfPas'][loc])
    parK1 = np.array(Pdict['Patch']['k1'][loc])
    parSfAct = np.array(Pdict['Patch']['SfAct'][loc])
    parVWALL = np.array(Pdict['Patch']['VWall'][loc])
    parAmRef = np.array(Pdict['Patch']['AmRef'][loc])

    t = Pdict['t']
    
    # idx0
    if idx0=='onsetQRS':
        idxED = model_instance.onsetQRS()
        
        # todo: improve
        if np.isnan(idxED):
            V = Pdict['Cavity']['V'][:,6]
            idxED = np.argmax(V)
    elif idx0=='PVcorner':
        p = Pdict['Node']['p'][:,6]
        V = Pdict['Cavity']['V'][:,6]

        #plt.plot(p,V)
        #plt.plot(p[:-2],np.diff(V,2)*1e3)

        idxED = np.argmax(np.diff(p,2)[:100])
        #plt.scatter(p[idxED],V[idxED])
        #plt.sho
    try:
        Ls = Pdict['Patch']['Ls'][idxED,loc]
    except:
        Ls = Pdict['Patch']['Ls'][idxED,loc]

    #return
    ret = []

    if what in ['contractility', 'dSfDtSfMax']:
        for iP in range(len(loc)):
            maxSfAct=np.max(Sf[:,iP]-SfPasT[:,iP])
            argmaxSfAct=np.argmax(Sf[:,iP]-SfPasT[:,iP])
    
            # if segment is non-contractile:
            idxStart = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] > 0.25*maxSfAct)
            idxEnd = np.argwhere(Sf[:argmaxSfAct,iP]-SfPasT[:argmaxSfAct,iP] < 0.75*maxSfAct)
            if len(idxStart)==0 or len(idxEnd)==0:
                ret.append(0)
            else:
                idxStart = idxStart[0,0]
                idxEnd = idxEnd[-1,-1]
        
                if idxEnd-idxStart<2:
                    ret.append(0)
                else:
                    try:
                        z = np.polyfit(t[idxStart:idxEnd].transpose()[0], Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP], 1)
                    except:
                        z = np.polyfit(t[idxStart:idxEnd].transpose()[0], Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP], 1)
                    z = np.polyfit(t[idxStart:idxEnd].transpose()[0], Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP], 1)
                    p = np.poly1d(z)
                    if what == 'dSfDtSfMax':
                        ret.append(z[0] / np.max(Sf[idxStart:idxEnd,iP] - SfPasT[idxStart:idxEnd,iP]))
                    else:
                        ret.append(np.array(z[0]))
    elif what in ['stiffness','compliance','DADTpass']:
        idxED = int(np.round(np.min((Pdict['Patch']['ActivationDelay']+Pdict['Patch']['dT'])[2:]/0.002)))
    
        if idx0=='onsetQRS':
            idxED = model_instance.onsetQRS()
        elif idx0=='PVcorner':
            V = Pdict['Node']['p'][:,6]
            p = Pdict['Cavity']['V'][:,6]
    
            #plt.plot(p,V)
            #plt.plot(p[:-2],np.diff(V,2)*1e3)
    
            idxED = np.argmax(np.diff(V,2)[:100])
            #plt.scatter(p[idxED],V[idxED])
            #plt.show()
    
        kk3 = 2 * parLsRef / pardLsPas
        LfP = Ls / parLs0Pas
        y = LfP**parK1
        yTit = LfP**kk3
        DSfPasDEf = y * (0.0349*parSfPas*parK1) + yTit * (0.01*parSfAct*kk3);
        SfEcm = (y - 1)*(0.0349*parSfPas);
        SfPasT = SfEcm+(yTit - 1)*(0.01*parSfAct);
        ls0  = np.log(Ls/parLs0Pas) - SfPasT/DSfPasDEf
        if what=='DADTpass':
            return Am[idxED,:]**2 / (DSfPasDEf - 2 * SfPasT) / (0.25*VWall)
        if what=='stiffness':
            return DSfPasDEf
        return 1/DSfPasDEf
    elif what == 'work':
        Ef1 = np.concatenate((Ef, Ef[[0],:]), axis=0)
        dEf = np.diff(Ef1, axis=0)
        return -np.sum((Sf * dEf), axis=0)
    return ret


def getCavity(model_instance,what,loc, idx0='onsetQRS'):
    Pdict = model_instance.getPdict()

    p = Pdict['Node']['p'][:,loc]
    V = Pdict['Cavity']['V'][:,loc]

    t = Pdict['t']
    

    #return
    ret = []

    if what == 'dpdtmax':
        ret = np.max(np.diff(p, axis=0) / (t[1]-t[0]), axis=0)
        #plt.figure(1).clf()
        #plt.subplot(2,1,1)
        #plt.plot(p*0.00750061683)
        #plt.subplot(2,1,2)
        #plt.plot(np.diff(p, axis=0) / (t[1]-t[0])*0.00750061683)
        
        

    return ret





if __name__ == "__main__":
    import Model
    model = Model.Model('../../CaSaPe/CaSaPe/CircAdapt.dll')
    model_instance = model.getInstance()
    model_instance.run()
    #print(get(model_instance,'contractility',np.array([2,3,4,5,6])))
    #print(get(model_instance,'compliance',np.array([2,3,4,5,6])))
    #print(get(model_instance,'stiffness',np.array([2,3,4,5,6])))
    
    print(getCavity(model_instance,'dpdtmax',np.array([6,7])))
