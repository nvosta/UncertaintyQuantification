# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import os, sys
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge, Rectangle

import _stiffness as ST

#from _Gauche1 import gauge
from _Gauche2 import gauge


#### COlors
RV_edge_color = [[0.2,0.2,0.2],
                 [0.2,0.2,0.2],
                 [0.2,0.2,0.2]]
RV_fill_color = [[135/255, 195/255, 91/255],
                 [34/255, 113/255, 181/255],
                 [236/255, 45/255, 36/255],]
LV_edge_color = [[0.2,0.2,0.2],
                 [0.2,0.2,0.2]]
LV_fill_color = [[60/255, 132/255, 63/255],
                 [217/255, 145/255, 9/255]]

Global_edge_color = [LV_edge_color[0],LV_edge_color[1],RV_edge_color[2]]
Global_fill_color = [LV_fill_color[0],LV_fill_color[1],RV_fill_color[2]]
    
    

def plot_dashboard(ais):
    pass




def get_dummy_pdicts(ais, sampler):
    Pdicts = []
    strain = []
    
    for theta in sampler.dummy_samples:
        model_instance = ais.model.getInstance()
        model_instance.setPdict(ais.prior_data['theta_info']['PdictOpt'])
        model_instance.setScalar('', '', '', 'tCycle', ais.patient.RVtime[-1])
        theta0 = ais.parameters.getX(model_instance)
        ais.parameters.setX(model_instance, theta)
        model_instance.run()
        
        if model_instance.getIsStable():
            try:
                strain.append(strain.append(ais.cost.getModelStrainIDX0corrected(model_instance,patient=ais.patient)[0]))
                Pdicts.append(model_instance.getPdict())
            except:
                pass
    return Pdicts, strain

def get_dummy_datadict(ais, sampler, n_samples = 100):
    Theta = []
    parVal = []
    
    strain = []
    
    SfAct = []
    k1 = []
    dT = []
    
    Sf = []
    T = []
    Ls = []
    
    contractility = []
    dSfDtSfMax = []
    stiffness = []
    dpdtmax = []
    DADTpass = []
    
    pressureLV = []
    pressureRV = []
    volumeLV = []
    volumeRV = []
    
    XmRatio = []
    
    RVD = []
    
    try:
        samples = sampler.dummy_samples
    except:
        samples = sampler.get_samples(n_samples)
    
    i=0
    while i < n_samples:
        print(i)
        theta=sampler.get_samples(1)[0]
        
        model_instance = ais.model.getInstance()
        #model_instance.setScalar('', '', '', 'tCycle', ais.patient.RVtime[-1])
        #ais.parameters.setX(model_instance, theta)
        #model_instance.run()
        model_instance = ais.model.getInstance()
        PdictRef = ais.prior_data['theta_info']['PdictOpt']
        if PdictRef is not None and PdictRef != []:
            model_instance.setPdict(PdictRef)
        model_instance.setScalar('','','','tCycle',np.round(ais.patient.RVtime[-1]/0.002)*0.002)
        theta0 = ais.parameters.getX(model_instance)
        for f in np.linspace(0,1,10):
            ais.parameters.setX(model_instance, f*theta0+(1-f)*theta)
            model_instance.runFast()  
            if np.any(np.isnan(model_instance.getVector('Rv','Patch','Rv1','Sf'))):
                print(' isnan')
                break
        if model_instance.getIsStable():
            ais.parameters.setX(model_instance, theta)
            model_instance.run()  
        elif not np.any(np.isnan(model_instance.getVector('Rv','Patch','Rv1','Sf'))):
                model_instance.run()  
        
        
        iTry = 10
        while iTry>0 and model_instance.getIsStable()==False and np.any(np.isnan(model_instance.getVector('Rv','Patch','Rv1','Sf')))==False:
            print('Model not hemodynamically stable')
            model_instance.run()
            iTry=iTry-1
        
        
        
        if model_instance.getIsStable()==False:
            print('Model crashed')
                
            
        Theta.append(theta);
        parVal.append(ais.parameters.XtoParVal(np.array(theta)))
        
        
        isGetStable = model_instance.getIsStable()
        #isGetStable = not np.any(np.isnan(model_instance.getVector('Lv','Cavity','','p')))
        if isGetStable:
            i+=1
            
            #try:
            s = ais.cost.getModelStrainIDX0corrected(model_instance,patient=ais.patient)
            if len(s)==2:
                s = s[0]
            strain.append(s)

            contractility.append(ST.get(model_instance, 'contractility',[4,5,6,3,2]))
            dSfDtSfMax.append(ST.get(model_instance, 'dSfDtSfMax',[4,5,6,3,2]))
            
            stiffness.append(ST.get(model_instance, 'stiffness',[4,5,6,3,2]))
            dpdtmax.append(ST.getCavity(model_instance, 'dpdtmax',[6,7]))
            DADTpass.append(ST.get(model_instance, 'DADTpass',[4,5,6,3,2]))
            
            
            pressureLV.append(model_instance.getVector('Lv','Cavity','','p'))
            pressureRV.append(model_instance.getVector('Rv','Cavity','','p'))
            volumeLV.append(model_instance.getVector('Lv','Cavity','','V'))
            volumeRV.append(model_instance.getVector('Rv','Cavity','','V'))
            
            Sf.append(np.array([model_instance.getVector('Rv','Patch','Rv1','Sf'),
                          model_instance.getVector('Rv','Patch','Rv2','Sf'),
                          model_instance.getVector('Rv','Patch','Rv3','Sf'),
                          model_instance.getVector('Sv','Patch','Sv1','Sf'),
                          model_instance.getVector('Lv','Patch','Lv1','Sf')]).transpose())
            T.append(np.array([model_instance.getVector('Rv','Patch','Rv1','T'),
                          model_instance.getVector('Rv','Patch','Rv2','T'),
                          model_instance.getVector('Rv','Patch','Rv3','T'),
                          model_instance.getVector('Sv','Patch','Sv1','T'),
                          model_instance.getVector('Lv','Patch','Lv1','T')]).transpose())
            Ls.append(np.array([model_instance.getVector('Rv','Patch','Rv1','Ls'),
                          model_instance.getVector('Rv','Patch','Rv2','Ls'),
                          model_instance.getVector('Rv','Patch','Rv3','Ls'),
                          model_instance.getVector('Sv','Patch','Sv1','Ls'),
                          model_instance.getVector('Lv','Patch','Lv1','Ls')]).transpose())
            
            Pdict = model_instance.getPdict()
            Am = Pdict['Wall']['Am'][:,6:9]
            Ym = Pdict['TriSeg']['Y']
            signCm = np.sign(Pdict['Wall']['Cm'][:,6:9])
            
            Xm = signCm * np.sqrt(Am/np.pi - Ym**2)
            XmRatio.append((Xm[:,2]-Xm[:,1]) / (Xm[:,1]-Xm[:,0]))
            RVD.append((Xm[:,2]-Xm[:,1]))
            
            # parameters
            SfAct.append([model_instance.getScalar('Rv','Patch','Rv1','SfAct'),
                          model_instance.getScalar('Rv','Patch','Rv2','SfAct'),
                          model_instance.getScalar('Rv','Patch','Rv3','SfAct'),
                          model_instance.getScalar('Sv','Patch','Sv1','SfAct'),
                          model_instance.getScalar('Lv','Patch','Lv1','SfAct')])
            k1.append([model_instance.getScalar('Rv','Patch','Rv1','k1'),
                          model_instance.getScalar('Rv','Patch','Rv2','k1'),
                          model_instance.getScalar('Rv','Patch','Rv3','k1'),
                          model_instance.getScalar('Sv','Patch','Sv1','k1'),
                          model_instance.getScalar('Lv','Patch','Lv1','k1')])
            dT.append([model_instance.getScalar('Rv','Patch','Rv1','dT'),
                          model_instance.getScalar('Rv','Patch','Rv2','dT'),
                          model_instance.getScalar('Rv','Patch','Rv3','dT'),
                          model_instance.getScalar('Sv','Patch','Sv1','dT'),
                          model_instance.getScalar('Lv','Patch','Lv1','dT')])
            
            print('succes')
                
            #except:
            #    pass
    return {'theta': Theta, 'parVal': parVal, 'strain':np.array(strain), 'SfAct': np.array(SfAct), 'k1': np.array(k1), 'dT': np.array(dT),
            'contractility':np.array(contractility),'dSfDtSfMax': np.array(dSfDtSfMax), 'stiffness':np.array(stiffness), 'dpdtmax':np.array(dpdtmax),
            'DADTpass':np.array(DADTpass),
            'pressureLV':np.array(pressureLV), 'pressureRV':np.array(pressureRV), 'volumeLV':np.array(volumeLV), 'volumeRV':np.array(volumeRV),
            'Sf':np.array(Sf),'T':np.array(T), 'Ls':np.array(Ls), 'XmRatio': np.array(XmRatio),
            'RVD':RVD}


def plot_dashboard_strain_measmod(ais, strain):
    plt.clf()
    ax = []
    ax.append(plt.subplot(2,2,1))
    plot_strain_meas_LV(ais.patient)
    plt.ylabel('Strain [%]')
    plt.title('LV')
    ax.append(plt.subplot(2,2,2))
    plot_strain_meas_RV(ais.patient)
    plt.title('RV')
    ax.append(plt.subplot(2,2,3))
    plot_strain_mod_LV(ais.patient, strain[:,:, 3:])
    plt.xlabel('Time [ms]')
    plt.ylabel('Strain [%]')
    ax.append(plt.subplot(2,2,4))
    plot_strain_mod_RV(ais.patient, strain[:,:, :3])
    plt.xlabel('Time [ms]')
    
    xMin = np.min([a.get_xlim()[0] for a in ax])
    xMax = np.max([a.get_xlim()[1] for a in ax])
    yMin = np.min([a.get_ylim()[0] for a in ax])
    yMax = np.max([a.get_ylim()[1] for a in ax])
    
    for a in ax:
        a.set_xlim([xMin, xMax])
        a.set_ylim([yMin, yMax])
        
def plot_dashboard_strainrate_measmod(ais, strain):
    plt.clf()
    ax = []
    ax.append(plt.subplot(2,2,1))
    plot_strainrate_meas_LV(ais.patient)
    plt.ylabel('Strain [%]')
    plt.title('LV')
    ax.append(plt.subplot(2,2,2))
    plot_strainrate_meas_RV(ais.patient)
    plt.title('RV')
    ax.append(plt.subplot(2,2,3))
    plot_strainrate_mod_LV(ais.patient, strain[:,:, 3:])
    plt.xlabel('Time [ms]')
    plt.ylabel('Strain [%]')
    ax.append(plt.subplot(2,2,4))
    plot_strainrate_mod_RV(ais.patient, strain[:,:, :3])
    plt.xlabel('Time [ms]')
    
    xMin = np.min([a.get_xlim()[0] for a in ax])
    xMax = np.max([a.get_xlim()[1] for a in ax])
    yMin = np.min([a.get_ylim()[0] for a in ax])
    yMax = np.max([a.get_ylim()[1] for a in ax])
    
    for a in ax:
        a.set_xlim([xMin, xMax])
        a.set_ylim([yMin, yMax])    
    

strainYLIM = [-42, 12]
def plot_strain_meas_LV(patient):
    xl = [0, np.max([1000, patient.RVtime[-1]*1e3])]
    plt.plot(xl, [0,0], '--', c=[0.5, 0.5, 0.5])
    
    plt.plot(patient.RVtime*1e3, patient.SVstrain, c=LV_fill_color[0])
    plt.plot(patient.RVtime*1e3, patient.LVstrain, c=LV_fill_color[1])
    # design settings
    plt.gca().set_facecolor((0,0,0))
    plt.xlim(xl)
    plt.ylim(strainYLIM)
    
def plot_strain_meas_RV(patient):
    xl = [0, np.max([1000, patient.RVtime[-1]*1e3])]
    plt.plot(xl, [0,0], '--', c=[0.5, 0.5, 0.5])
    
    for i in range(3):
        plt.plot(patient.RVtime*1e3, patient.RVstrain[:,i], c=RV_fill_color[i])
    # design settings
    plt.gca().set_facecolor((0,0,0))
    plt.xlim(xl)
    plt.ylim(strainYLIM)
    
def plot_strain_mod_LV(patient, strain):
    xl = [0, np.max([1000, patient.RVtime[-1]*1e3])]
    plt.plot(xl, [0,0], '--', c=[0.5, 0.5, 0.5])
    time = patient.RVtime
    plot_mod_strain(time, strain, LV_fill_color)
    plt.xlim(xl)
    plt.ylim(strainYLIM)
    
def plot_strain_mod_RV(patient, strain):
    xl = [0, np.max([1000, patient.RVtime[-1]*1e3])]
    plt.plot(xl, [0,0], '--', c=[0.5, 0.5, 0.5])
    time = patient.RVtime
    plot_mod_strain(time, strain, RV_fill_color)
    plt.xlim(xl)
    plt.ylim(strainYLIM)
    

def plot_strainrate_meas_LV(patient):
    xl = [0, np.max([1000, patient.RVtime[-1]*1e3])]
    plt.plot(xl, [0,0], '--', c=[0.5, 0.5, 0.5])
    
    plt.plot(patient.RVtime[:-1]*1e3, np.diff(patient.SVstrain), c=LV_fill_color[0])
    plt.plot(patient.RVtime[:-1]*1e3, np.diff(patient.LVstrain), c=LV_fill_color[1])
    # design settings
    plt.gca().set_facecolor((0,0,0))
    plt.xlim(xl)
    
def plot_strainrate_meas_RV(patient):
    xl = [0, np.max([1000, patient.RVtime[-1]*1e3])]
    plt.plot(xl, [0,0], '--', c=[0.5, 0.5, 0.5])
    
    for i in range(3):
        plt.plot(patient.RVtime[:-1]*1e3, np.diff(patient.RVstrain[:,i]), c=RV_fill_color[i])
    # design settings
    plt.gca().set_facecolor((0,0,0))
    plt.xlim(xl)
    
def plot_strainrate_mod_LV(patient, strain):
    xl = [0, np.max([1000, patient.RVtime[-1]*1e3])]
    plt.plot(xl, [0,0], '--', c=[0.5, 0.5, 0.5])
    time = patient.RVtime
    print(strain.shape)
    print(np.diff(strain).shape)
    plot_mod_strain(time[:-1], np.diff(strain, axis=1), LV_fill_color)
    plt.xlim(xl)
def plot_strainrate_mod_RV(patient, strain):
    xl = [0, np.max([1000, patient.RVtime[-1]*1e3])]
    plt.plot(xl, [0,0], '--', c=[0.5, 0.5, 0.5])
    time = patient.RVtime
    plot_mod_strain(time[:-1], np.diff(strain, axis=1), RV_fill_color)
    plt.xlim(xl)



def plot_mod_strain(time, strain, color):
    minstrain = 100
    maxstrain = -100
    for i in range(len(strain)):
        for j in range(len(strain[i])):
            minstrain = np.min([minstrain, np.min(strain[i][j])])
            maxstrain = np.max([maxstrain, np.max(strain[i][j])])
    
    
    
    for i in range(len(color)): # range(5):
        s = []
        minLen = 999999
        for j in range(len(strain)):
            minLen = np.min([minLen, strain[j][:,i].shape[0]  ])
        for j in range(len(strain)):
            s.append(strain[j][:minLen,i])
        s = np.array(s).transpose()
        t = 2*np.array(range(s.shape[0]))
        print(len(t))
        
        ss1 = np.sort(s, axis=1)
        print(ss1.shape)
        nSamples = 100
        ss = np.zeros((nSamples, ss1.shape[1]))
        
        x = np.linspace(0,1,nSamples)
        xp = np.linspace(0,1,ss1.shape[0])
    
        for iLine in range(ss1.shape[1]):
            ss[:,iLine] = np.interp(x, xp, ss1[:,iLine])
        
        print(len(t), len(x), len(xp))
        t = np.interp(x,xp, t)
        
        
        plotRanges = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        plotRanges = np.array([0.025, 0.5, 0.975])
        
        interval_idx = np.floor(plotRanges * s.shape[1]).astype(int)
        interval_idx[interval_idx>ss.shape[1]-2] = ss.shape[1]-1
            
        plt.fill_between(t, ss[:,interval_idx[0]], ss[:,interval_idx[-1]], alpha=0.2, color=color[i])
        #plt.fill_between(t, ss[:,interval_idx[0]], ss[:,interval_idx[-1]], hatch='', fc=[0,0,0,0], ec=np.append(color[i], 0.5))
        #plt.fill_between(t, ss[:,interval_idx[1]], ss[:,interval_idx[-2]], alpha=0.1, color=[0.5,0.5,0.5])
        #plt.plot(t, ss[:,interval_idx[np.array([0,1,3,4])]], linewidth=1, color=color[i])
        plt.plot(t, ss[:,interval_idx[np.array([0,2])]], '-', linewidth=1, color=color[i], alpha=0.1)
        #plt.plot(t, ss[:,interval_idx[2]], linewidth=2, color=color[i])
        plt.plot(t, ss[:,interval_idx[1]], linewidth=2, color=color[i])
        
        
        

        dstrain = (np.max(ss[:,interval_idx[np.array([0,2])]])-np.min(ss[:,interval_idx[np.array([0,2])]]))*0.25
        plt.ylim([np.min(ss[:,interval_idx[np.array([0,2])]])-dstrain, np.max(ss[:,interval_idx[np.array([0,2])]])+dstrain])
        
        



#&&=========================
def plot_dashboard_hemodynamic_vectors(ais, dummy_datadict):
    plt.clf()
    m,n = 4,3
    
    time = np.linspace(0, ais.patient.RVtime[-1], dummy_datadict['pressureRV'].shape[1])
    
    colorLV=LV_fill_color[0]
    colorRV=LV_fill_color[1]
    
    label_pressure = 'Pressure [mmHg]'
    label_volume = 'Volume [mL]'
    label_stress = 'Stress [kPa]'
    label_tension = 'Tension [N]'
    label_ls = 'Sarcomere Length [\mu m]'
    label_time = 'Time [ms]'
    label_nat_strain = 'ef [-]'

    plt.subplot(m,n,3)
    plot_distribution_vector(time, dummy_datadict['pressureRV']*0.00750061683, colorRV)
    plot_distribution_vector(time, dummy_datadict['pressureLV']*0.00750061683, colorLV)
    plt.xlabel(label_time)
    plt.ylabel(label_pressure)

    plt.subplot(m,n,6)
    plot_distribution_vector(time, dummy_datadict['volumeRV']*1e6, colorRV)
    plot_distribution_vector(time, dummy_datadict['volumeLV']*1e6, colorLV)
    plt.xlabel(label_time)
    plt.ylabel(label_volume)
    
    plt.subplot(m,n,9)
    plot_distribution_vector(dummy_datadict['volumeRV']*1e6, dummy_datadict['pressureRV']*0.00750061683, colorRV)
    plot_distribution_vector(dummy_datadict['volumeLV']*1e6, dummy_datadict['pressureLV']*0.00750061683, colorLV)
    plt.xlabel(label_volume)
    plt.ylabel(label_pressure)
    
    plt.subplot(m,n,1)
    #plot_distribution_vector(time, dummy_datadict['Sf'][:,0], colorRV)
    plot_mod_strain(time, dummy_datadict['T'][:,:,:3]*1e-3, RV_fill_color)
    plt.xlabel(label_time)
    plt.ylabel(label_tension)
    
    plt.subplot(m,n,4)
    #plot_distribution_vector(time, dummy_datadict['Sf'][:,0], colorRV)
    plot_mod_strain(time, dummy_datadict['Sf'][:,:,:3]*1e-3, RV_fill_color)
    plt.xlabel(label_time)
    plt.ylabel(label_stress)
    
    plt.subplot(m,n,7)
    #plot_distribution_vector(time, dummy_datadict['Sf'][:,0], colorRV)
    plot_mod_strain(time, dummy_datadict['Ls'][:,:,:3], RV_fill_color)
    plt.xlabel(label_time)
    plt.ylabel(label_ls)
    
    plt.subplot(m,n,10)
    plot_distribution_vector(dummy_datadict['Ls'][:,:,0], dummy_datadict['Sf'][:,:,0]*1e-3, RV_fill_color[0])
    plot_distribution_vector(dummy_datadict['Ls'][:,:,1], dummy_datadict['Sf'][:,:,1]*1e-3, RV_fill_color[1])
    plot_distribution_vector(dummy_datadict['Ls'][:,:,2], dummy_datadict['Sf'][:,:,2]*1e-3, RV_fill_color[2])
    plt.xlabel(label_ls)
    plt.ylabel(label_stress)

    

    plt.subplot(m,n,2)
    #plot_distribution_vector(time, dummy_datadict['Sf'][:,0], colorRV)
    plot_mod_strain(time, dummy_datadict['T'][:,:,3:]*1e-3, LV_fill_color)
    plt.xlabel(label_time)
    plt.ylabel(label_tension)
    
    plt.subplot(m,n,5)
    #plot_distribution_vector(time, dummy_datadict['Sf'][:,0], colorRV)
    plot_mod_strain(time, dummy_datadict['Sf'][:,:,3:]*1e-3, LV_fill_color)
    plt.xlabel(label_time)
    plt.ylabel(label_stress)
    
    plt.subplot(m,n,8)
    #plot_distribution_vector(time, dummy_datadict['Sf'][:,0], colorRV)
    plot_mod_strain(time, dummy_datadict['Ls'][:,:,3:], LV_fill_color)
    plt.xlabel(label_time)
    plt.ylabel(label_ls)
    
    plt.subplot(m,n,11)
    plot_distribution_vector(dummy_datadict['Ls'][:,:,3], dummy_datadict['Sf'][:,:,3]*1e-3, LV_fill_color[0])
    plot_distribution_vector(dummy_datadict['Ls'][:,:,4], dummy_datadict['Sf'][:,:,4]*1e-3, LV_fill_color[1])
    plt.xlabel(label_ls)
    plt.ylabel(label_stress)
    
    plt.subplot(m,n,12)
    plot_distribution_vector(time, np.array(dummy_datadict['RVD'])*1e3, LV_fill_color[0])
    plt.xlabel(label_ls)
    plt.ylabel('RV dimension [mm]')
    
    


def plot_distribution_vector(X, Y, color):
    for i in range(len(Y)):
        if len(X)==len(Y):
            plt.plot(X[i], Y[i], c=color, alpha=0.2, linewidth=1)
        else:
            #plt.plot(X, Y[i], c=color)
            
            s = []
            minLen = 999999
            for j in range(len(Y)):
                minLen = np.min([minLen, Y[j].shape[0]  ])
            for j in range(len(Y)):
                s.append(Y[j][:minLen])
            s = np.array(s).transpose()
            
            ss = np.sort(s, axis=1)
            plotRanges = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
            plotRanges = np.array([0.025, 0.5, 0.975])
            
            interval_idx = np.floor(plotRanges * s.shape[1]).astype(int)
            interval_idx[interval_idx>ss.shape[1]-2] = ss.shape[1]-1
                
            plt.fill_between(X, ss[:,interval_idx[0]], ss[:,interval_idx[-1]], alpha=0.05, color=[0.5,0.5,0.5])
            #plt.fill_between(t, ss[:,interval_idx[1]], ss[:,interval_idx[-2]], alpha=0.1, color=[0.5,0.5,0.5])
            #plt.plot(t, ss[:,interval_idx[np.array([0,1,3,4])]], linewidth=1, color=color[i])
            plt.plot(X, ss[:,interval_idx[np.array([0,2])]], '--', linewidth=1, color=color, alpha=0.5)
            #plt.plot(t, ss[:,interval_idx[2]], linewidth=2, color=color[i])
            plt.plot(X, ss[:,interval_idx[1]], linewidth=2, color=color)




####################################################################
# Dashboard gauge

def plot_dashboard_gauge_parameters_RV(ais, dummy_datadict):
    plt.clf()
    m,n = 4,6
    ax = []
    ax.append(plt.subplot(m,n,1))
    parval = dummy_datadict['SfAct'][:,:3]
    maxVal = np.ceil( np.max(parval) / 120000)
    maxVal = np.max([maxVal, 800])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=120000, rangeMin=minVal, rangeMax=maxVal*1200,
                                    labels=["%0.f" % (i) for i in np.linspace(minVal, maxVal, 5)],
                                    title='SfAct [%]')
    ax.append(plt.subplot(m,n,2))
    parval = dummy_datadict['SfAct'][:,3:]
    maxVal = np.ceil( np.max(parval) / 120000)
    maxVal = np.max([maxVal, 800])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=120000, rangeMin=minVal, rangeMax=maxVal*1200,
                                    labels=["%0.f" % (i) for i in np.linspace(minVal, maxVal, 5)],
                                    title='SfAct [%]', colors=LV_fill_color[::-1])
    
    ax.append(plt.subplot(m,n,3))
    parval = dummy_datadict['k1'][:,:3]
    maxVal = np.ceil( np.max(parval) / 10)
    maxVal = np.max([maxVal, 1000])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=120000, rangeMin=minVal, rangeMax=maxVal/10,
                                    labels=["%0.f" % (i) for i in np.linspace(minVal, maxVal, 5)],
                                    title='k1 [%]')
    ax.append(plt.subplot(m,n,4))
    parval = dummy_datadict['k1'][:,3:]
    maxVal = np.ceil( np.max(parval) / 10)
    maxVal = np.max([maxVal, 1000])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=120000, rangeMin=minVal, rangeMax=maxVal/10,
                                    labels=["%0.f" % (i) for i in np.linspace(minVal, maxVal, 5)],
                                    title='k1 [%]', colors=LV_fill_color[::-1])
    
    ax.append(plt.subplot(m,n,5))
    #plot_dashboard_gauge_parameter(ais, (dummy_datadict['dT'][:,:3]-np.mean(dummy_datadict['dT'][:,:3], axis=1).reshape(-1,1))*1e3,
    plot_dashboard_gauge_parameter(ais, dummy_datadict['dT'][:,:3]*1e3,
                                    logscale=False, refValue=0, rangeMin=-100, rangeMax=100,
                                    labels=['-100','-50','0','50','100'],
                                    title='dT [ms]' )
    
    ax.append(plt.subplot(m,n,6))
    #plot_dashboard_gauge_parameter(ais, (dummy_datadict['dT'][:,:3]-np.mean(dummy_datadict['dT'][:,:3], axis=1).reshape(-1,1))*1e3,
    plot_dashboard_gauge_parameter(ais, dummy_datadict['dT'][:,3:]*1e3,
                                    logscale=False, refValue=0, rangeMin=-100, rangeMax=100,
                                    labels=['-100','-50','0','50','100'],
                                    title='dT [ms]', colors=LV_fill_color[::-1] )
    
    
    ax.append(plt.subplot(m,n,7))
    parval = dummy_datadict['contractility'][:,:3]*1e-3
    maxVal = np.ceil( np.max(parval) / 100) * 100
    maxVal = np.max([maxVal, 1000])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                    labels=["%0.f" % (i) for i in np.linspace(minVal, maxVal, 5)],
                                    title='dSfDt [kPa/s]')
    ax.append(plt.subplot(m,n,8))
    parval = dummy_datadict['contractility'][:,3:]*1e-3
    maxVal = np.ceil( np.max(parval) / 100) * 100
    maxVal = np.max([maxVal, 1000])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                    labels=["%0.f" % (i) for i in np.linspace(minVal, maxVal, 5)],
                                    title='dSfDt [kPa/s]', colors=LV_fill_color[::-1])
    
    ax.append(plt.subplot(m,n,9))
    parval = dummy_datadict['stiffness'][:,:3]*1e-3
    maxVal = np.ceil( np.max(parval) / 500) * 500
    maxVal = np.max([maxVal, 5000])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                    labels=["%0.f" % (i) for i in np.linspace(minVal, maxVal, 5)],
                                    title='DSfPasDEf [kPa]')
    ax.append(plt.subplot(m,n,10))
    parval = dummy_datadict['stiffness'][:,3:]*1e-3
    maxVal = np.ceil( np.max(parval) / 500) * 500
    maxVal = np.max([maxVal, 5000])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                    labels=["%0.f" % (i) for i in np.linspace(minVal, maxVal, 5)],
                                    title='DSfPasDEf [kPa]', colors=LV_fill_color[::-1])
   
    ax.append(plt.subplot(m,n,11))
    #plot_dashboard_gauge_parameter(ais, (dummy_datadict['dT'][:,:3]-np.mean(dummy_datadict['dT'][:,:3], axis=1).reshape(-1,1))*1e3,
    plot_dashboard_gauge_parameter(ais, (dummy_datadict['dT'][:,:3]-np.mean(dummy_datadict['dT'], axis=1).reshape(-1,1))*1e3,
                                    logscale=False, refValue=0, rangeMin=-100, rangeMax=100,
                                    labels=['-100','-50','0','50','100'],
                                    title='dT-mean(dT) [ms]' )
    
    ax.append(plt.subplot(m,n,12))
    #plot_dashboard_gauge_parameter(ais, (dummy_datadict['dT'][:,:3]-np.mean(dummy_datadict['dT'][:,:3], axis=1).reshape(-1,1))*1e3,
    plot_dashboard_gauge_parameter(ais, (dummy_datadict['dT'][:,3:]-np.mean(dummy_datadict['dT'], axis=1).reshape(-1,1))*1e3,
                                    logscale=False, refValue=0, rangeMin=-100, rangeMax=100,
                                    labels=['-100','-50','0','50','100'],
                                    title='dT-mean(dT) [ms]', colors=LV_fill_color[::-1] )
    
    
    
    ax.append(plt.subplot(m,n,4*6))
    plot_dashboard_gauge_parameter(ais, dummy_datadict['dpdtmax']*0.00750061683,
                                    logscale=False, refValue=1, rangeMin=0, rangeMax=1600,
                                    labels=['0','400','800','1200', '1600'],
                                    title='dpdtmax [mmHg/s]')
    
    
    
    if 'dSfDtSfMax' in dummy_datadict:
        ax.append(plt.subplot(m,n,13))
        parval = dummy_datadict['dSfDtSfMax'][:,:3]
        maxVal = np.ceil( np.max(parval) / 5 ) * 5
        maxVal = np.max([maxVal, 25])
        minVal = 0
        plot_dashboard_gauge_parameter(ais, parval,
                                        logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                        labels=[str(i) for i in np.linspace(minVal, maxVal, 3)],
                                        title='dSfDt/SfMax [s^-1]')
        ax.append(plt.subplot(m,n,14))
        parval = dummy_datadict['dSfDtSfMax'][:,3:]
        maxVal = np.ceil( np.max(parval) / 5 ) * 5
        maxVal = np.max([maxVal, 25])
        minVal = 0
        plot_dashboard_gauge_parameter(ais, parval,
                                        logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                        labels=[str(i) for i in np.linspace(minVal, maxVal, 3)],
                                        title='dSfDt/SfMax [s^-1]', colors=LV_fill_color[::-1])
    
    ax.append(plt.subplot(m,n,15))
    parval = dummy_datadict['DADTpass'][:,:3]*1e4
    maxVal = np.ceil( np.max(parval) / 1 ) * 1
    maxVal = np.max([maxVal, 4])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                    labels=[str(i) for i in np.linspace(minVal, maxVal, 3)],
                                    title='DADTpass [cm^2/N]')
    
    ax.append(plt.subplot(m,n,16))
    parval = dummy_datadict['DADTpass'][:,3:]*1e4
    maxVal = np.ceil( np.max(parval) / 1 ) * 1
    maxVal = np.max([maxVal, 4])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                    labels=[str(i) for i in np.linspace(minVal, maxVal, 3)],
                                    title='DADTpass [cm^2/N]', colors=LV_fill_color[::-1])
    
    ax.append(plt.subplot(m,n,17))
    #plot_dashboard_gauge_parameter(ais, (dummy_datadict['dT'][:,:3]-np.mean(dummy_datadict['dT'][:,:3], axis=1).reshape(-1,1))*1e3,
    plot_dashboard_gauge_parameter(ais, np.mean(dummy_datadict['dT'], axis=1).reshape(-1,1)*1e3,
                                    logscale=False, refValue=0, rangeMin=-100, rangeMax=100,
                                    labels=['-100','-50','0','50','100'],
                                    title='mean(dT) [ms]', colors=LV_fill_color[::-1] )
    
    if 'dSfDtSfMax' in dummy_datadict:
        ax.append(plt.subplot(m,n,19))
        parval = dummy_datadict['dSfDtSfMax'][:,:3]/np.max(dummy_datadict['dSfDtSfMax'][:,:3], axis=1).reshape(-1,1)*100
        maxVal = np.ceil( np.max(parval) / 5 ) * 5
        maxVal = np.max([maxVal, 25])
        minVal = 0
        plot_dashboard_gauge_parameter(ais, parval,
                                        logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                        labels=[str(i) for i in np.linspace(minVal, maxVal, 3)],
                                        title='dSfDt/SfMax [% resp. to max]')
        ax.append(plt.subplot(m,n,20))
        parval = dummy_datadict['dSfDtSfMax'][:,3:]/np.max(dummy_datadict['dSfDtSfMax'][:,3:], axis=1).reshape(-1,1)*100
        maxVal = np.ceil( np.max(parval) / 5 ) * 5
        maxVal = np.max([maxVal, 25])
        minVal = 0
        plot_dashboard_gauge_parameter(ais, parval,
                                        logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                        labels=[str(i) for i in np.linspace(minVal, maxVal, 3)],
                                        title='dSfDt/SfMax [% resp. to max]', colors=LV_fill_color[::-1])
    
    ax.append(plt.subplot(m,n,21))
    parval = dummy_datadict['DADTpass'][:,:3]/np.min(dummy_datadict['DADTpass'][:,:3], axis=1).reshape(-1,1)*100
    maxVal = np.ceil( np.max(parval) / 100 ) * 100
    maxVal = np.max([maxVal, 5])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                    labels=[str(i) for i in np.linspace(minVal, maxVal, 3)],
                                    title='DADTpass [% resp. to min]')
    ax.append(plt.subplot(m,n,22))
    parval = dummy_datadict['DADTpass'][:,3:]/np.min(dummy_datadict['DADTpass'][:,3:], axis=1).reshape(-1,1)*100
    maxVal = np.ceil( np.max(parval) / 100 ) * 100
    maxVal = np.max([maxVal, 5])
    minVal = 0
    plot_dashboard_gauge_parameter(ais, parval,
                                    logscale=False, refValue=1, rangeMin=minVal, rangeMax=maxVal,
                                    labels=[str(i) for i in np.linspace(minVal, maxVal, 3)],
                                    title='DADTpass [% resp. to min]', colors=LV_fill_color[::-1])
    
   
    
    xMin = np.min([a.get_xlim()[0] for a in ax])
    xMax = np.max([a.get_xlim()[1] for a in ax])
    yMin = np.min([a.get_ylim()[0] for a in ax])
    yMax = np.max([a.get_ylim()[1] for a in ax])
    
    plt.subplots_adjust(0.04, 0.01, 0.96, 0.85, 0.2, 0)
    
    for a in ax:
        a.set_xlim([xMin, xMax])
        a.set_ylim([yMin, yMax])
    

def plot_dashboard_gauge_parameter(ais, parVal, logscale=True, refValue=1, rangeMin = -3, rangeMax = 1, CI = 0.95, CI1=0.5, title='', colors=None, labels=[]):
    plt.gca().cla()
    
    if colors is None:
        colors=RV_fill_color
    
    nRange = len(labels)
   
    if logscale:
        meanParval = np.log2(np.median(parVal, axis=0))
        minVal = np.log2(refValue)+ rangeMin
        maxVal = np.log2(refValue)+ rangeMax
    
        arrow = ( meanParval - minVal ) #/ (maxVal - minVal) 
    
        sortParval = np.sort(parVal, axis=0)
        
        uncertainty_left = np.log2(sortParval[int( (1-CI)/2  *parVal.shape[0]),:]) - minVal
        uncertainty_right = np.log2(sortParval[int( (1-(1-CI)/2) *parVal.shape[0]),:]) - minVal
        
        uncertainty_left1 = np.log2(sortParval[int( (1-CI1)/2  *parVal.shape[0]),:]) - minVal
        uncertainty_right1 = np.log2(sortParval[int( (1-(1-CI1)/2) *parVal.shape[0]),:]) - minVal
        
        uncertainty_left[np.isnan(uncertainty_left)] = 0
        uncertainty_right[np.isnan(uncertainty_right)] = maxVal
        uncertainty_left1[np.isnan(uncertainty_left1)] = 0
        uncertainty_right1[np.isnan(uncertainty_right1)] = maxVal
    
        if len(labels)==0:
            labels = 2**np.linspace(rangeMin, rangeMax, nRange)
    else:
        meanParval = np.median(parVal, axis=0)
        
        minVal = rangeMin # - (rangeMax-rangeMin)/10
        maxVal = rangeMax #+ (rangeMax-rangeMin)/10
        
        arrow = ( meanParval - minVal ) / (maxVal - minVal) * nRange
        
        sortParval = np.sort(parVal, axis=0)
        
        uncertainty_left = sortParval[int( (1-CI)/2  *parVal.shape[0]),:]
        uncertainty_right = sortParval[int( (1-(1-CI)/2) *parVal.shape[0]),:]
        uncertainty_left1 = sortParval[int( (1-CI1)/2  *parVal.shape[0]),:]
        uncertainty_right1 = sortParval[int( (1-(1-CI1)/2) *parVal.shape[0]),:]
        
        uncertainty_left = ( uncertainty_left - minVal ) / (maxVal - minVal) * nRange
        uncertainty_right = ( uncertainty_right - minVal ) / (maxVal - minVal) * nRange
        uncertainty_left1 = ( uncertainty_left1 - minVal ) / (maxVal - minVal) * nRange
        uncertainty_right1 = ( uncertainty_right1 - minVal ) / (maxVal - minVal) * nRange
                
        if len(labels)==0:
            labels = np.linspace(rangeMin, rangeMax, nRange)
        
    
    gauge(plt.gca(), labels=labels, \
      arrowColor=colors, 
      arrow=arrow/nRange, 
      uncertainty_left=uncertainty_left/nRange, 
      uncertainty_right=uncertainty_right/nRange, 
      uncertainty_left1=uncertainty_left1/nRange, 
      uncertainty_right1=uncertainty_right1/nRange, 
      logscale=logscale,
      title=title) 
