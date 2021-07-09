# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
sys.path.insert(1, 'src/')
import _stiffness as ST
import plot_adaptive_importance_sampling_dashboard as paisd

from _violin import violin

import Patient
import Model

import addcopyfighandler
import matplotlib
matplotlib.rcParams['savefig.format'] = 'svg'

folder = 'data/final_AMIS/'

patient = 2

dummydicts = []
patients = []
fiew_folder_name = ['']
    
#fn = folder+'vII202005_'+str(patient)+'_F1/dummy_datadict.npy'
fn = folder+'dummy_datadict_vII202005_'+str(patient)+'_F1.npy'
if os.path.isfile(fn):
    dummydicts.append(np.load(fn, allow_pickle=True).item())
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_'+str(patient)+'_F1.npy'))
    
    # add work
    Sf = dummydicts[-1]['Sf']
    Ef = np.log(dummydicts[-1]['Ls']/2)
    Ef = np.concatenate((Ef, Ef[:,[0],:]), axis=1)
    dEf = np.diff(Ef, axis=1)
    
    dummydicts[-1]['work'] = -np.sum((Sf * dEf), axis=1)

print('len dummydicts: ', len(dummydicts))
iView_time = range(len(dummydicts))

#dummydicts=[dummydicts[0], dummydicts[0]]

if patient==61:
    iView_time = [0,3.471232877,4.517808219,7.728767123,9.079452055]

#iView_time = [0, 2]
iView_time=np.array(iView_time)
iView_strainplot = [0, int((len(dummydicts)-1)/2), len(dummydicts)-1]


if len(dummydicts)==0:
    raise Exception('No data')

#%%

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


#%%
if False:
    #%%
    plt.figure(1)
    plt.clf()
    for iPl in range(7):
        plt.subplot(3,3,iPl+1)
        color = RV_fill_color
        const=1
        ylabel=None
        if iPl==0:
            struct_label='SfAct'
            ylabel = 'SfAct [kPa]'
            norm=0
            i_locs = np.array([0,1,2])
            const=1e-3
        elif iPl==1:
            struct_label='k1'
            ylabel = 'k1 [-]'
            norm=0
            i_locs = np.array([0,1,2])
        elif iPl==2:
            struct_label='dT'
            ylabel = 'dT [ms]'
            norm=0
            const=1e3
            i_locs = np.array([0,1,2])
            const = 1e3
        
        elif iPl==3:
            struct_label='SfAct'
            ylabel = 'SfAct [kPa]'
            norm=0
            i_locs = np.array([3,4])
            const=1e-3
            color=LV_fill_color
        elif iPl==4:
            struct_label='k1'
            ylabel = 'k1 [-]'
            norm=0
            i_locs = np.array([3,4])
            color=LV_fill_color
        elif iPl==5:
            struct_label='dT'
            ylabel = 'dT [ms]'
            norm=0
            const=1e3
            i_locs = np.array([3,4])
            const = 1e3
            color=LV_fill_color
            
        elif iPl==6:
            struct_label='AmRef'
            ylabel = 'AmRef [cm^2]'
            norm=0
            const=1e3
            i_locs = np.array([0,1,2])
            const = 1e3
            color=np.concatenate((LV_fill_color, RV_fill_color))
            
            
            
        elif False:
            
                
            if iPl==3:
                struct_label='contractility'
                ylabel = 'Contractility [kPa/s]'
                norm=0
                const=1e-3
                i_locs = np.array([0,1,2])
            elif iPl==4:
                struct_label='DADTpass'
                ylabel='Compliance [cm2/N]'
                norm=0
                const=1e4
                i_locs = np.array([0,1,2])
            elif iPl==5:
                struct_label='work'
                ylabel='Work density [kPa]'
                norm=0
                const=1e-3
                i_locs = np.array([0,1,2])
            elif iPl==6:
                struct_label='contractility'
                norm=1
                i_locs = np.array([0,1,2])
            elif iPl==7:
                struct_label='stiffness'
                norm=2
                i_locs = np.array([0,1,2])
            elif iPl==8:
                struct_label='DADTpass'
                norm=1
                i_locs = np.array([0,1,2])
        else:
            if True:
                struct_label='dSfDtSfMax'
                struct_label='contractility'
            if True:
                struct_label='DADTpass'
                struct_label='stiffness'
            
            i_locs = np.array([0,1,2])
            color = RV_fill_color
            norm=0
        
        if ylabel is None:
            ylabel = struct_label
        
        confidence_intervals = [[] for _ in range(len(i_locs))]
        
        
        plt.ylabel(ylabel)
        for i_view in range(len(dummydicts)):
            for i_loc in range(len(i_locs)):
                Y = dummydicts[i_view][struct_label][:, i_loc]*const
                if norm==1:
                    Y=Y/np.max(dummydicts[i_view][struct_label][:, i_locs], axis=1)
                if norm==2:
                    Y=Y/np.min(dummydicts[i_view][struct_label][:, i_locs], axis=1)
                if norm==3:
                    Y=Y-np.mean(dummydicts[i_view][struct_label][:, i_locs], axis=1)
                #plt.scatter(np.ones(Y.shape)*i_view-0.2+i_loc*0.2, Y, c=color[i_loc])
                confidence_intervals[i_loc].append(np.percentile(Y, [2.5, 25, 50, 75, 97.5]))
        
        confidence_intervals = np.array(confidence_intervals)
        
        
        
        
        style = 2
        for i_loc in range(len(i_locs)):
            if style==0:
                plt.plot(iView_time, confidence_intervals[i_loc,:,0],'--', linewidth=1, color=color[i_loc])
                plt.plot(iView_time, confidence_intervals[i_loc,:,1],'--', linewidth=1, color=color[i_loc])
                plt.plot(iView_time, confidence_intervals[i_loc,:,2], '-', linewidth=2, color=color[i_loc])
                plt.plot(iView_time, confidence_intervals[i_loc,:,3],'--', linewidth=1, color=color[i_loc])
                plt.plot(iView_time, confidence_intervals[i_loc,:,4],'--', linewidth=1, color=color[i_loc])
                
                plt.fill_between(
                    iView_time, 
                    confidence_intervals[i_loc,:,0], 
                    confidence_intervals[i_loc,:,4], alpha=0.05, color=[0.5,0.5,0.5])
                
                plt.fill_between(
                    iView_time, 
                    confidence_intervals[i_loc,:,1], 
                    confidence_intervals[i_loc,:,3], alpha=0.25, color=[0.5,0.5,0.5])
            elif style==1:
                #plt.plot(iView_time, confidence_intervals[i_loc,:,2], '-', linewidth=2, color=color[i_loc])
                plt.scatter(iView_time-0.25+0.25*i_loc, confidence_intervals[i_loc,:,2], ec=color[i_loc], fc=[1,1,1],zorder=99)
    #            plt.scatter(iView_time, confidence_intervals[i_loc,:,2], ec=color[i_loc], fc=[1,1,1],zorder=99)
                
                for i_view in range(confidence_intervals.shape[1]):
                    
                    allLocConf = [[0,4],[1,3]]
                    allAlpha = [0.25, 1]
                    for locConf, alpha in zip(allLocConf, allAlpha):
                        
                        x = iView_time[i_view]-0.25+0.25*i_loc
                        plt.fill_between(x+np.array([-0.125,0.125]),
                                     confidence_intervals[i_loc,i_view,0]*np.ones(2),
                                     confidence_intervals[i_loc,i_view,4]*np.ones(2),
                                     fc=color[i_loc],
                                     alpha=0.2
                                     )
                        plt.fill_between(x+np.array([-0.125,0.125]),
                                     confidence_intervals[i_loc,i_view,1]*np.ones(2),
                                     confidence_intervals[i_loc,i_view,3]*np.ones(2),
                                     fc=color[i_loc],
                                     alpha=1
                                     )
                    #plt.xticks(range(12), ['' for _ in range(12)])
                    plt.xticks([])
            elif style==2:
                #plt.plot(iView_time, confidence_intervals[i_loc,:,2], '-', linewidth=2, color=color[i_loc])
                #plt.scatter(iView_time-0.25+0.25*i_loc, confidence_intervals[i_loc,:,2], ec=color[i_loc], fc=[1,1,1],zorder=99)
    #            plt.scatter(iView_time, confidence_intervals[i_loc,:,2], ec=color[i_loc], fc=[1,1,1],zorder=99)
                
                for i_view in range(confidence_intervals.shape[1]):
                    
                    allLocConf = [[0,4],[1,3]]
                    allAlpha = [0.25, 1]
                    for locConf, alpha in zip(allLocConf, allAlpha):
                        
                        bw = 0.75/len(i_locs)
                        x = iView_time[i_view]+bw*i_loc - 0.5*bw*(len(i_locs)-1)
                        print(i_view, i_loc, x)
                        plt.plot([x-0.01,x-0.01], 
                             confidence_intervals[i_loc,i_view,np.array([0, 4])], 
                             color=color[i_loc],
                             linewidth=1,
                             zorder = 1)
                        
                        plt.plot([x+np.array([-bw*1/3,-bw*1/3]),x+np.array([bw*1/3,bw*1/3])], 
                                 confidence_intervals[i_loc,i_view,np.array([[0, 4],[0,4]])], 
                                 color=color[i_loc],
                                 linewidth=2,
                                 zorder = 1)
                        
                        plt.plot(x+np.array([-bw*1/3,bw*1/3]), 
                                 confidence_intervals[i_loc,i_view,np.array([2,2])], 
                                 color=[1,1,1],
                                 linewidth=1,
                                 zorder = 1)
                        plt.fill_between(x+np.array([-0.5*bw,0.5*bw]),
                                     confidence_intervals[i_loc,i_view,1]*np.ones(2),
                                     confidence_intervals[i_loc,i_view,3]*np.ones(2),
                                     fc=color[i_loc],
                                     alpha=1
                                     )
                    #plt.xticks(range(12), ['' for _ in range(12)])
                    plt.xticks([])
    # background
        from matplotlib.colors import ListedColormap
        
        xl = plt.xlim()
        xl = [0-0.5,12-0.5]
        plt.xlim(xl)
        yl = plt.ylim()
        yl = np.array(yl) + np.array([-0.05, 0.2])*np.diff(yl)
        plt.ylim(yl)
    
        plt.fill_between([xl[0], 2.5], yl[0], yl[1],
                                     fc=[0.5,0.5,0.5],
                                     alpha=0.1
                                     )
        plt.fill_between([5.5, 8.5], yl[0], yl[1],
                                     fc=[0.5,0.5,0.5],
                                     alpha=0.1
                                     )
        
        # intra observer text
        plt.plot(xl, [yl[1]-np.diff(yl)*0.1, yl[1]-np.diff(yl)*0.1], c=[0.7,0.7,0.7], linewidth=1)
        plt.plot(xl, [yl[1]-np.diff(yl)*0.15, yl[1]-np.diff(yl)*0.15], c=[0.7,0.7,0.7], linewidth=1)
        plt.plot([5.5,5.5], [yl[0], yl[1]-np.diff(yl)*0.05], c=[0.7,0.7,0.7], linewidth=2)
        
        #white line
        for xloc in [2.5, 8.5]:
            plt.plot([xloc,xloc], [yl[0], yl[1]-np.diff(yl)*0.1], c=[0.7,0.7,0.7], linewidth=1)
        
        
        
        plt.fill_between([xl[0], 5.5], yl[1]-np.diff(yl)*0.1, yl[1]-np.diff(yl)*0.05,
                                     fc=[0.7,0.7,0.7],
                                     alpha=1
                                     )
        
        plt.text(1, yl[1]-np.diff(yl)*0.1, 'Intra-observer')
        
        plt.fill_between([5.5, xl[1]], yl[1]-np.diff(yl)*0.1, yl[1]-np.diff(yl)*0.05,
                                     fc=[1,1,1],
                                     alpha=1
                                     )
        plt.text(7, yl[1]-np.diff(yl)*0.1, 'Intra-observer')
        
        
        plt.fill_between([xl[0], xl[1]], yl[1]-np.diff(yl)*0.05, yl[1],
                                     fc=[0.2,0.2,0.2],
                                     alpha=1
                                     )
        plt.text(4, yl[1]-np.diff(yl)*0.045, 'Inter-observer', color=[0.95,0.95,0.95])
        
        
        for xLoc in [0.5, 3.7, 6.7, 10]:
            plt.text(xLoc, yl[1]-np.diff(yl)*0.15, 'AIS')
        
        
        
        
    
    
    
#%%
if False:
    fig = plt.figure(2, figsize=(12,9))
    plt.clf()
    for iiView in range(len(iView_strainplot)):
        iView = iView_strainplot[iiView]
        plt.subplot(3,4,1+4*iiView)
        paisd.plot_strain_meas_LV(patients[iView])
        plt.ylabel('Strain [%]')
        plt.title('LV')
        plt.subplot(3,4,3+4*iiView)
        paisd.plot_strain_meas_RV(patients[iView])
        plt.title('RV')
        strain = dummydicts[iView]['strain']
        plt.subplot(3,4,2+4*iiView)
        paisd.plot_strain_mod_LV(patients[iView], strain[:,:, 3:])
        plt.xlabel('Time [ms]')
        plt.ylabel('Strain [%]')
        plt.subplot(3,4,4+4*iiView)
        paisd.plot_strain_mod_RV(patients[iView], strain[:,:, :3])
        plt.xlabel('Time [ms]')
    
#%%

def plotSingleBars(dummydict, struct_label='SfAct', ref=None, color = RV_fill_color, norm=0, const=1):
  
    i_locs = np.array([0,1,2])
    confidence_intervals = [[] for _ in range(len(i_locs))]
    
    Y_iloc = []
    
    for i_loc in range(len(i_locs)):
        Y = dummydict[struct_label][:, i_loc]
        if norm==1:
            Y=Y/np.max(dummydict[struct_label][:, i_locs], axis=1)
        if norm==2:
            Y=Y/np.min(dummydict[struct_label][:, i_locs], axis=1)
        if norm==3:
            Y=Y-np.mean(dummydict[struct_label][:, i_locs], axis=1)
        #plt.scatter(np.ones(Y.shape)*i_view-0.2+i_loc*0.2, Y, c=color[i_loc])
        confidence_intervals[i_loc].append(np.percentile(Y, [2.5, 25, 50, 75, 97.5]))
        Y_iloc.append(Y * const)
    
    confidence_intervals = np.array(confidence_intervals) * const
    
    for i_loc in range(3):
        #plt.plot(i_loc+np.array([-0.5,0.5])*1, confidence_intervals[i_loc,:,2]*np.ones(2), c='w', zorder=99)
        #plt.scatter(i_loc, confidence_intervals[i_loc,:,2], ec=color[i_loc], fc=[1,1,1],zorder=99, s=100)
        ax = plt.gca()
        for i_view in range(confidence_intervals.shape[1]):
            
            allLocConf = [[0,4],[1,3]]
            allAlpha = [0.25, 1]
            for locConf, alpha in zip(allLocConf, allAlpha):
                style = 2
                if style==0:
                    plt.fill_between(i_loc+np.array([-0.5,0.5]),
                                     confidence_intervals[i_loc,i_view,0]*np.ones(2),
                                     confidence_intervals[i_loc,i_view,4]*np.ones(2),
                                     fc=color[i_loc],
                                     ec=None,
                                     alpha=0.2
                                     )
                    plt.fill_between(i_loc+np.array([-0.5,0.5]),
                                     confidence_intervals[i_loc,i_view,0]*np.ones(2),
                                     confidence_intervals[i_loc,i_view,4]*np.ones(2),
                                     fc=[0,0,0,0],
                                     ec=[0.5,0.5,0.5]
                                     )
                    plt.fill_between(i_loc+np.array([-0.5,0.5]),
                                     confidence_intervals[i_loc,i_view,1]*np.ones(2),
                                     confidence_intervals[i_loc,i_view,3]*np.ones(2),
                                     fc=color[i_loc],
                                     alpha=1
                                     )
                elif style==1:
                    plt.plot([i_loc,i_loc], 
                             confidence_intervals[i_loc,i_view,np.array([0, 4])], 
                             color=color[i_loc],
                             linewidth=2,
                             zorder = 1)
                    plt.plot([i_loc+np.array([-1/3,-1/3]),i_loc+np.array([1/3,1/3])], 
                             confidence_intervals[i_loc,i_view,np.array([[0, 4],[0,4]])], 
                             color=color[i_loc],
                             linewidth=2,
                             zorder = 1)
                    plt.fill_between(i_loc+np.array([-0.5,0.5]),
                                     confidence_intervals[i_loc,i_view,1]*np.ones(2),
                                     confidence_intervals[i_loc,i_view,3]*np.ones(2),
                                     fc=color[i_loc],
                                     alpha=1, 
                                     zorder = 2
                                     )
                elif style==2:
                    violin(i_loc, Y_iloc[i_loc], np.ones(len(Y_iloc[i_loc])),
                           percentile=[],
                           c=color[i_loc], left=0.5, right=0.5)
                
                
                    
                    if ref is not None:
                        plt.scatter(i_loc,ref[i_loc]*const, c=[0,0,0], zorder=3)
                elif style==2:

                    l=0.15
                    r=0.15
                    
                    violin(i_loc, 
                           Y_iloc[i_loc], 
                           np.ones(Y_iloc[i_loc].shape), 
                           c=color[i_loc],
                           left=l, right=r, percentile=[])
                    if ref is not None:
                        plt.scatter(i_loc,ref[i_loc]*const, c=[0,0,0], zorder=3)
                    
                
    plt.xlim([-1,2.75])
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.xticks([0,1,2],['Apex', 'Mid','Base'], rotation=45)
    


if True:
    plotIdataset = 0
    
    strain = dummydicts[plotIdataset]['strain']
    
    # creating grid for subplots
    fig = plt.figure(3)
    plt.clf()
    fig.set_figheight(7)
    fig.set_figwidth(12)
    
    plt.subplots_adjust(top=0.966,
    bottom=0.1,
    left=0.056,
    right=0.968,
    hspace=0.20,
    wspace=4)
    
    
    Sf = dummydicts[plotIdataset]['Sf']
    Ef = np.log(dummydicts[plotIdataset]['Ls']/2)
    Ef = np.concatenate((Ef, Ef[:,[0],:]), axis=1)
    dEf = np.diff(Ef, axis=1)
    
    dummydicts[plotIdataset]['work'] = -np.sum((Sf * dEf), axis=1)
    
    Pdict = patients[0].modelPdict
      
    ax1 = plt.subplot2grid(shape=(2, 15), loc=(0, 0), colspan=4, rowspan=1)
    paisd.plot_strain_meas_LV(patients[plotIdataset])
    plt.title('LV strain')
    ax2 = plt.subplot2grid(shape=(2, 15), loc=(0, 4), colspan=4, rowspan=1)
    paisd.plot_strain_meas_RV(patients[plotIdataset])
    plt.title('RV strain')
    ax3 = plt.subplot2grid(shape=(2, 15), loc=(0, 9), colspan=2, rowspan=1)
    plotSingleBars(dummydicts[plotIdataset], ref=Pdict['Patch']['SfAct'][4:], const=1e-3)
    plt.ylabel('SfAct [kPa]', labelpad=2)
    
    ax4 = plt.subplot2grid(shape=(2, 15), loc=(0, 11), colspan=2, rowspan=1)
    plotSingleBars(dummydicts[plotIdataset], ref=Pdict['Patch']['k1'][4:],struct_label='k1', const=1)
    plt.ylabel('k1 [-]', labelpad=2)
    
    ax5 = plt.subplot2grid(shape=(2, 15), loc=(0, 13), colspan=2, rowspan=1)
    plotSingleBars(dummydicts[plotIdataset], ref=Pdict['Patch']['dT'][4:],struct_label='dT', const=1e3)
    plt.ylabel('dT [ms]', labelpad=-5)
    
    
    model = Model.Model('src/CircAdapt.dll')
    model_instance = model.getInstance()
    model_instance.setPdict(Pdict)
    model_instance.run()
    
    ax6 = plt.subplot2grid(shape=(2, 15), loc=(1, 0), colspan=4, rowspan=1)
    paisd.plot_strain_mod_LV(patients[plotIdataset], strain[:,:, 3:])
    ax7 = plt.subplot2grid(shape=(2, 15), loc=(1, 4), colspan=4, rowspan=1)
    paisd.plot_strain_mod_RV(patients[plotIdataset], strain[:,:, :3])
    ax8 = plt.subplot2grid(shape=(2, 15), loc=(1, 9), colspan=2, rowspan=1)
    plotSingleBars(dummydicts[plotIdataset], 
                   struct_label='contractility', 
                   ref=ST.get(model_instance, 'contractility',[4,5,6]),
                   const=1e-3)
    plt.ylabel('Contractility [kPa/s]', labelpad=0)
    ax9 = plt.subplot2grid(shape=(2, 15), loc=(1, 11), colspan=2, rowspan=1)
    plotSingleBars(dummydicts[plotIdataset], 
                   struct_label='DADTpass', 
                   ref=ST.get(model_instance, 'DADTpass',[4,5,6]),
                   const=1e4)
    plt.ylabel('Compliance [cm2/N]', labelpad=0)
    ax10 = plt.subplot2grid(shape=(2, 15), loc=(1, 13), colspan=2, rowspan=1)
    plotSingleBars(dummydicts[plotIdataset], 
                   struct_label='work', 
                   ref=ST.get(model_instance, 'work',[4,5,6]),
                   const=1e-3)
    plt.ylabel('work density [kPa]', labelpad=0)
    
      
    # display plot
    plt.show()