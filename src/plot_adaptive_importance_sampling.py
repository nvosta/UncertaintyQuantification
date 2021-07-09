import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.neighbors import KernelDensity
import scipy.stats as sst
import os

def plot_single_importance_sampling(ais, s=None, i_iter=0, plot='parameters', plotIQR=False, pointType='scatter', indexRange = range(15),
                                    equalXaxis=[], plotBar=True, iFig=1, XRef = None):
    thetaRef = XRef
    use_iters = range(i_iter)
    if plot == 'indices':
        theta = np.array(ais.prior_data['theta_info']['indices'][:,indexRange])
        X2 = np.array(ais.prior_data['theta_info']['X2'])
    else:
        theta = np.array(ais.prior_data['theta_info']['theta'])
        X2 = np.array(ais.prior_data['theta_info']['X2'])



    print('min X2: ', np.min(X2[np.isnan(X2) == False]))
    print('nSamples: ', len(X2))

    plotSubs = range(theta.shape[1])
    if plot=='parameters':
        nPar = ais.parameters.XtoParVal(np.array(theta)).shape[1]
        plotSubs = range(nPar)
        
    n, m = 5, 4
    figsize = (16, 9)
    if len(plotSubs) == 1:
        n,m = 1, 1
        figsize = (5, 5)
    elif len(plotSubs) <= 5:
        n,m = len(plotSubs), 1
    elif len(plotSubs) <= 10:
        n,m = 5, 2
    elif len(plotSubs) <= 15:
        n, m = 5, 3
    elif len(plotSubs) <= 16:
        n, m = 4, 4
    elif len(plotSubs) <= 20:
        n, m = 5, 4
    elif len(plotSubs) <= 25:
        n, m = 5, 5

    ax = [[] for iPar in plotSubs ]
    iPlot = -1
    for iPar in plotSubs:
        if iPar+1-iPlot*m*n > m*n:
            iPlot = iPlot + 1
            plt.figure(iFig, figsize=figsize)
            plt.clf()
            iFig+=1
        ax[iPar] = plt.subplot(m, n, iPar+1-iPlot*m*n)
        if len(plotSubs) > 1:
            plt.subplots_adjust(top=0.959,
                bottom = 0.065,
                left = 0.048,
                right = 0.964,
                hspace = 0.573,
                wspace = 0.603)
        else:
            plt.subplots_adjust(top=0.9,
                bottom=0.1,
                left=0.2,
                right=0.8)

        if plot=='theta':
            parval = theta[:,iPar]
            try:
                parvalRef = np.array(thetaRef)[iPar]
            except:
                pass
        elif plot=='parameters':
            parval = ais.parameters.XtoParVal(np.array(theta))[:,iPar] / ais.parameters.parameterGetScale(iPar=iPar)
            try:
                parvalRef = ais.parameters.XtoParVal(np.array(thetaRef).reshape(1,-1))[0][iPar] / ais.parameters.parameterGetScale(iPar=iPar)
            except:
                pass
        else:
            parval = theta[:,iPar]

        #if pointType=='scatter':

        plt.yscale('log')
        #elif pointType=='hist':



        plt.scatter(parval, X2, ec=[0,0,0],fc=[1,1,1],s=2)


        if plot=='parameters':
            plt.xlabel(ais.parameters.parameterGetLabelModel(iPar=iPar,includeLocation=True))
            plt.title(ais.parameters.parameterGetLabelClinical(iPar=iPar))


        elif plot=='indices':
            plt.xlabel(ais.options['costSaveIndices'].getName(iPar))
        elif plot=='theta':
            try:
                plt.xlabel(ais.parameters.thetaGetLable(iPar=iPar,includeLocation=True))
                plt.title(ais.parameters.thetaGetTitle(iPar=iPar))
            except:
                pass
        plt.ylabel(r'$\chi^2$')

        ax[iPar].twinx()
        if s is not None:
            if plot=='indices':
                pass #plot_probability_density_single_parameter_approach(parval, ais, iPar,plot=plot, plotIQR=plotIQR)
            else:
                plot_probability_density_single_parameter(s, ais, iPar,plot=plot, plotIQR=plotIQR)
            plt.ylabel(r'$p(\theta | x)$')

            yl = plt.ylim()







        if False:# plotBar:
            nBins = int(np.max([5,np.min([np.sqrt(n_eff), 1000])]))
             # determine nbins for this theta
            asTheta = np.argsort(theta[:,iPar])
            cssw = np.cumsum(sample_weight[asTheta])

            t0 = np.min(theta[:,iPar])
            t05 = theta[ asTheta[len(cssw)-(cssw<0.05*cssw[-1])[::-1].argmax()-1] , iPar]
            t95 = theta[ asTheta[(cssw>0.95*cssw[-1])[::1].argmax()] , iPar]
            t100 = np.max(theta[:,iPar])

            nBins1 = int( nBins * 0.9 * (t100-t0) / np.max([0.01*(t100-t0), (t95-t05)]) )

            hi,bi = np.histogram(parval, weights=sample_weight, bins=nBins1)
            hi1,bi1 = np.histogram(parval, bins=bi)

            hi = hi / np.sum(hi) * len(hi) / (bi[-1]-bi[0])
            width = np.diff(bi[:2])
            plt.bar(bi[:-1]+0.5*width, hi, width=width,
                 color=[0.5,0.5,0.5,0.8], zorder=1)

            plt.ylim(yl)

        try:
            parvalRef = parvalRef
            plt.scatter(parvalRef, 0, s=100, ec=[0,0,0],fc=[1,1,1], zorder=2)
        except:
            pass

    if plot=='parameters':
        if len(ax)>15:
            setSameXLim(ax[:5])
            setSameXLim(ax[5:10])
            setSameXLim(ax[10:15])
    if len(equalXaxis)>0:
        for ea in equalXaxis:
            setSameXLim(np.array(ax)[np.array(ea)])


def setSameXLim(ax):
    xl = [np.Infinity, -np.Infinity]
    for ax0 in ax:
        xl0 = ax0.get_xlim()
        xl[0] = np.min([xl[0], xl0[0]])
        xl[1] = np.max([xl[1], xl0[1]])
    for ax0 in ax:
        ax0.set_xlim(xl[0],xl[1])

def plot_probability_density_single_parameter(s, ais, iPar, plot='', plotIQR=False):
    dumm_samples = s.get_samples(1000)



    # go to parameter space
    if plot=='parameters':
        dumm_samples = ais.parameters.XtoParVal(np.array(dumm_samples))[:,iPar] / ais.parameters.parameterGetScale(iPar=iPar)
    else:
        dumm_samples = np.array(dumm_samples)[:,iPar]

    # handle boundaries
    stdS = np.std(dumm_samples)

    #xmin = np.min(dumm_samples)
    #xmax = np.max(dumm_samples)
    #xls = np.linspace(xmin,xmax,100)

    try:
        blablabla
        xls = s.get_important_points_for_plot_distribution(iPar)

        score = np.exp(s.score_samples_1D(xls,iPar))

        plt.plot(xls,score,c="k",linewidth=5)
        plt.plot(xls,score,c="w",linewidth=1)

        return
    except:
        pass


    dumm_samples = dumm_samples[np.isinf(dumm_samples)==False]
    dumm_samples = dumm_samples[np.isnan(dumm_samples)==False]
    stdS = np.std(dumm_samples)
    stdS = np.max([stdS, 1e-99])
    bandwidth = stdS*len(dumm_samples)**(-1/5)
    print(bandwidth)

    # fit kernel
    kde = KernelDensity(bandwidth=bandwidth).fit(dumm_samples.reshape(-1, 1))

    # plot
    ss = np.sort(dumm_samples)
    xmin,xmax = ss[int(len(ss)*0.05)]-3*stdS, ss[int(len(ss)*0.95)]+3*stdS
    #if xmin<0 and ais.parameters.out_of_bound(-1e-9, iPar):
    #    xmin=0
    x = np.linspace(xmin,xmax,1000)

    y = np.exp(kde.score_samples(x.reshape(-1, 1)))

    #for ix in range(len(x)):
    #    if ais.parameters.out_of_bound(x):
    #        y[ix]=0

    #y = y / np.sum(y) * len(y)
    plt.plot(x[10:-10],y[10:-10],c="k",linewidth=5)
    plt.plot(x,y,c="w",linewidth=1)
    plt.xlim([min(x),max(x)])

    if plotIQR:
        q0 = np.percentile(dumm_samples, 0)
        q1 = np.percentile(dumm_samples, 25)
        q2 = np.percentile(dumm_samples, 50)
        q3 = np.percentile(dumm_samples, 75)
        q4 = np.percentile(dumm_samples, 100)

        yl = [0, 0.1*np.max(y)]

        plt.plot([q0,q0], yl, 'k')
        plt.plot([q1,q1], yl, 'k')
        plt.plot([q2,q2], yl, 'k')
        plt.plot([q3,q3], yl, 'k')
        plt.plot([q4,q4], yl, 'k')


def plot_probability_density_single_parameter_approach(values, ais, iPar, plot='', plotIQR=False):


    X2 = ais.prior_data['theta_info']['X2']
    w = np.array(ais.prior_data['iter_info']['nSims'])/ np.sum(ais.prior_data['iter_info']['nSims'])
    w = w.reshape((1,-1))
    weight_prior = np.sum(w * np.exp(
        np.array(ais.prior_data['theta_info']['pi'])), axis=1)
    weight_prior[weight_prior<1e-300] = 1e-300
    weight_prior = np.log(weight_prior)


    stdS = np.sqrt(np.cov(values, aweights=np.exp(weight_prior)))

    bandwidth = stdS*len(values)**(-1/5)
    print(bandwidth)

    sample_weight = np.exp(-X2-weight_prior)
    sample_weight= sample_weight / np.sum(sample_weight) * 100

    # fit kernel
    kde = KernelDensity(bandwidth=bandwidth, kernel=ais.options['kdeSettings']['kernel']).fit(values.reshape(-1, 1), sample_weight=sample_weight)

    # plot
    ss = np.sort(values)
    xmin,xmax = ss[int(len(ss)*0.1)]-3*stdS, ss[int(len(ss)*0.9)]+3*stdS
    #if xmin<0 and ais.parameters.out_of_bound(-1e-9, iPar):
    #    xmin=0
    x = np.linspace(xmin,xmax,1000)

    y = np.exp(kde.score_samples(x.reshape(-1, 1)))

    #for ix in range(len(x)):
    #    if ais.parameters.out_of_bound(x):
    #        y[ix]=0

    #y = y / np.sum(y) * len(y)
    plt.plot(x[10:-10],y[10:-10],c="k",linewidth=5)
    plt.plot(x,y,c="w",linewidth=1)
    plt.xlim([min(x[y>0.001*np.max(y)]),max(x[y>0.001*np.max(y)])])

    if plotIQR:
        q0 = np.percentile(values, 0)
        q1 = np.percentile(values, 25)
        q2 = np.percentile(values, 50)
        q3 = np.percentile(values, 75)
        q4 = np.percentile(values, 100)

        yl = [0, 0.1*np.max(y)]

        plt.plot([q0,q0], yl, 'k')
        plt.plot([q1,q1], yl, 'k')
        plt.plot([q2,q2], yl, 'k')
        plt.plot([q3,q3], yl, 'k')
        plt.plot([q4,q4], yl, 'k')





def plot_load_samplers(samplerFiles,setNames):
    samplers = []
    for file in samplerFiles:
        samplers.append(np.load(file,allow_pickle=True))
        #print(samplers[-1])
        samplers[-1]['sampler'].load_dummy_samples(100, max_time=10)
        #print('sample:',samplers[-1]['sampler'].get_sample())
        #print('dummy_samples:',samplers[-1]['sampler'].dummy_samples.shape)
    return samplers

import pickle
def plot_load_priordata(priorFiles,setNames):
    priordata = []
    for file in priorFiles:
        priordata.append(np.load(file,allow_pickle=True).item())
    return priordata



def plot_boxplot_comparison(samplerFiles=[],setNames=[],samplers=[]):
    plt.figure(figsize=(16,9))
    plt.subplots_adjust(top=0.962,
        bottom=0.06,
        left=0.08,
        right=0.992,
        hspace=0.507,
        wspace=0.15)

    if samplers==[]:
        samplers = plot_load_samplers(samplerFiles, setNames)




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



    ax0 = plt.subplot(4,2,1)
    plot_boxplot_comparison_single_parameter(samplers,setNames, LV_edge_color, LV_fill_color, plotPar=[0,1])
    ax1 = plt.subplot(4,2,2)
    plot_boxplot_comparison_single_parameter(samplers,setNames, RV_edge_color, RV_fill_color, plotPar=[2,3,4])
    equalizeYlim(ax0,ax1)
    ax0 = plt.subplot(4,2,3)
    plot_boxplot_comparison_single_parameter(samplers,setNames, LV_edge_color, LV_fill_color, plotPar=[5,6])
    ax1 = plt.subplot(4,2,4)
    plot_boxplot_comparison_single_parameter(samplers,setNames, RV_edge_color, RV_fill_color, plotPar=[7,8,9])
    equalizeYlim(ax0,ax1)
    ax0 = plt.subplot(4,2,5)
    plot_boxplot_comparison_single_parameter(samplers,setNames, LV_edge_color, LV_fill_color, plotPar=[10,11])
    ax1 = plt.subplot(4,2,6)
    plot_boxplot_comparison_single_parameter(samplers,setNames, RV_edge_color, RV_fill_color, plotPar=[12,13,14])
    equalizeYlim(ax0,ax1)
    ax0 = plt.subplot(4,2,7)
    plot_boxplot_comparison_single_parameter(samplers,setNames, Global_edge_color, Global_fill_color, plotPar=[15,16,17])
    ax1 = plt.subplot(4,4,15)
    plot_boxplot_comparison_single_parameter(samplers,setNames, RV_edge_color, RV_fill_color, plotPar=[18])
    ax1 = plt.subplot(4,4,16)
    plot_boxplot_comparison_single_parameter(samplers,setNames, RV_edge_color, RV_fill_color, plotPar=[19])





def equalizeYlim(ax0,ax1):
    yl0 = ax0.get_ylim()
    yl1 = ax1.get_ylim()
    yl = [np.min([yl0[0], yl1[0]]),np.max([yl0[1], yl1[1]])]
    ax0.set_ylim(yl[0],yl[1])
    ax1.set_ylim(yl[0],yl[1])

def plot_boxplot_comparison_single_parameter(samplers,setNames, edge_color, fill_color, plotPar=[]):
    print(plotPar)
    data = [[] for i_d in range(len(plotPar))]
    for i_sampler in range(len(samplers)):
        d = []
        for i_d in range(len(plotPar)):
            if plotPar[i_d] < len(samplers[i_sampler]['parameters']):
                d0 = np.array(samplers[i_sampler]['sampler'].dummy_samples)
                d0 = (samplers[i_sampler]['parameters'].XtoParVal(d0)[:,plotPar[i_d]] /
                      samplers[i_sampler]['parameters'].parameterGetScale(iPar=plotPar[i_d]))
                data[i_d].append(d0)
            else:
                data[i_d].append(samplers[i_sampler]['sampler'].dummy_samples[:,0]*0+5.1)

    print('data shape:',len(data),len(data[0]))

    # --- Labels for your data:
    labels_list = setNames
    xlocations  = np.array(range(len(data[0])))
    space       = 0.1
    xlocations  = xlocations + space * xlocations
    width       = (1 - space) / len(data)
    print('width',width)
    symbol      = 'r+'
    ymin        = 0
    ymax        = 10

    ax = plt.gca()
    #ax.set_ylim(ymin,ymax)
    #ax.set_xticks( range(len(labels_list)) )
    ax.set_xticklabels( labels_list, rotation=0 )
    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.set_xticks(xlocations)
    plt.ylabel(
        samplers[i_sampler]['parameters'].parameterGetLabelModel(iPar=plotPar[0]) +
        ' ['+ samplers[i_sampler]['parameters'].parameterGetUnit(iPar=plotPar[0]) + ']')
    plt.title(samplers[i_sampler]['parameters'].parameterGetLabelClinical(iPar=plotPar[0]))

    # --- Offset the positions per group:
    positions_group = []
    for i_d in range(len(data)):
        positions_group.append([x+width*i_d-0.5+0.5*space for x in xlocations])

    bp=[[] for i_d in range(len(plotPar))]
    for i_d in range(len(data)):
        bp[i_d] = ax.boxplot(data[i_d],
                sym=symbol,
                labels=['']*len(labels_list),
                positions=positions_group[i_d],
                widths=width,
                notch=False,
                vert=True,
                whis=[2.5, 97.5],
                usermedians=None,
                conf_intervals=None,
                patch_artist=True,
                )
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[i_d][element], color=edge_color[i_d])

        for patch in bp[i_d]['boxes']:
            patch.set_facecolor(fill_color[i_d])

    bpleg = [bp[i]["boxes"][0] for i in range(len(bp))]
    legnames = []
    try:
        for i in  range(len(bp)):
            loc = samplers[i_sampler]['parameters'].parameters[plotPar[i]][2]
            if type(loc)==list:
                loc = loc[0][:2]
            legnames.append(loc)
    except:
        legnames = []
        for i in  range(len(bp)):
            if plotPar[i]<len(samplers[i_sampler]['parameters']):
                legnames.append(samplers[i_sampler]['parameters'].locnames[plotPar[i]])
            else:
                legnames.append('')

    ax.legend(bpleg, legnames, loc='upper right')
    xl = plt.xlim()
    #plt.xlim([xl[0],xl[1]+0.1*np.diff(xl)])



def plot_strain(ais, sampler, cost=[], plotFeature='Strain'):
    samplesModel, strain = get_strain_from_thetas(ais, sampler, cost, plotFeature=plotFeature)

    plot_strain_ranges(ais, sampler, strain, plotFeature=plotFeature)
    #plot_strain_pure(ais, sampler, strain)

def get_strain_from_thetas(ais, sampler,cost=[], plotFeature='Strain'):
    samplesModel = []
    strain = []
    for theta in sampler.dummy_samples:
        model_instance = ais.model.getInstance()
        model_instance.setScalar('', '', '', 'tCycle', ais.patient.RVtime[-1])
        ais.parameters.setX(model_instance, theta)
        model_instance.run()
        if model_instance.getIsStable():
            #if cost!=[]:
            #    cost.plotValidation(ais.patient, model_instance)
            samplesModel.append(model_instance.getPdict())
            if plotFeature=='Strain':
                try:
                    strain.append(ais.cost.getModelStrainIDX0corrected(model_instance,patient=ais.patient)[0])
                except:
                    pass
            else:
                sf = []
                sf.append(model_instance.getVector('Lv','Patch','Lv1',plotFeature))
                sf.append(model_instance.getVector('Sv','Patch','Sv1',plotFeature))
                sf.append(model_instance.getVector('Rv','Patch','Rv1',plotFeature))
                sf.append(model_instance.getVector('Rv','Patch','Rv2',plotFeature))
                sf.append(model_instance.getVector('Rv','Patch','Rv3',plotFeature))
                sf = np.array(sf).transpose()
                strain.append(sf)



    return samplesModel, strain

def plot_strain_ranges(ais, sampler, strain, plotFeature='Strain'):
    plt.figure()

    minstrain = 100
    maxstrain = -100
    for i in range(len(strain)):
        for j in range(len(strain[i])):
            minstrain = np.min([minstrain, np.min(strain[i][j])])
            maxstrain = np.max([maxstrain, np.max(strain[i][j])])



    for i in range(5):
        plt.subplot(2,3,i+1)
        s = []
        minLen = 999999
        for j in range(len(strain)):
            minLen = np.min([minLen, strain[j][:,i].shape[0]  ])
        for j in range(len(strain)):
            s.append(strain[j][:minLen,i])
        s = np.array(s).transpose()
        t = 2*np.array(range(s.shape[0]))

        ss = np.sort(s, axis=1)
        plotRanges = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

        interval_idx = np.floor(plotRanges * s.shape[1]).astype(int)
        interval_idx[interval_idx>ss.shape[1]-2] = ss.shape[1]-1

        plt.fill_between(t, ss[:,interval_idx[0]], ss[:,interval_idx[-1]], alpha=0.5, color=[0.5,0.5,0.5])
        plt.fill_between(t, ss[:,interval_idx[1]], ss[:,interval_idx[-2]], alpha=0.5, color=[0.5,0.5,0.5])
        plt.plot(t, ss[:,interval_idx[np.array([0,1,3,4])]], linewidth=2, color=[0.5,0.5,0.5])
        plt.plot(t, ss[:,interval_idx[2]], linewidth=1, color=[0,0,0])

        if i<3:
            plt.title(['Rv'+str(i+1)])
            if plotFeature=='Strain':
                plt.plot(ais.patient.RVtime*1e3, ais.patient.RVstrain[:,i], linewidth=3, color='k')
        elif i<4:
            plt.title('Sv')
            if plotFeature=='Strain':
                plt.plot(ais.patient.RVtime*1e3, ais.patient.SVstrain, linewidth=3, color='k')
        elif i<5:
            plt.title('Lv')
            if plotFeature=='Strain':
                plt.plot(ais.patient.RVtime*1e3, ais.patient.LVstrain, linewidth=3, color='k')

        dstrain = (maxstrain-minstrain)*0.05
        plt.ylim([minstrain-dstrain, maxstrain+dstrain])


    plt.show()

def plot_strain_pure(ais, sampler, strain):
    plt.figure()
    minstrain = 0
    maxstrain = 1
    for i in range(len(strain)):
        for j in range(len(strain[i])):
            minstrain = np.min([minstrain, np.min(strain[i][j])])
            maxstrain = np.max([maxstrain, np.max(strain[i][j])])
    for i in range(5):
        plt.subplot(2,3,i+1)
        s = []
        minLen = 999999
        for j in range(len(strain)):
            minLen = np.min([minLen, strain[j][:,i].shape[0]  ])
        for j in range(len(strain)):
            s.append(strain[j][:minLen,i])
        s = np.array(s).transpose()
        t = 2*np.array(range(s.shape[0]))


        for i1 in range(s.shape[1]):
            plt.plot(t, s[:,i1])

        if i<3:
            plt.plot(ais.patient.RVtime*1e3, ais.patient.RVstrain[:,i], linewidth=3, color='k')
        elif i<4:
            plt.plot(ais.patient.RVtime*1e3, ais.patient.SVstrain, linewidth=3, color='k')
        elif i<5:
            plt.plot(ais.patient.RVtime*1e3, ais.patient.LVstrain, linewidth=3, color='k')


        plt.ylim([minstrain-5, maxstrain+5])

    plt.show()



def plot_single_importance_sampling_all_iterations(ais, s, i_iter=1, plot='theta', final_sampler_settings=None, skipIter = 1):
    # collect data

    samples_sampler = []
    samples_final = []

    nSamples = 1000


    samples_sampler = np.ndarray((0, nSamples, len(ais.parameters)))

    filename = ais.options['saveFolder']+'plot_data_all_iterations.npy'
    if os.path.isfile(filename):
        data = np.load(filename, allow_pickle=True)
        if len(data[0]==nSamples):
            samples_sampler = data

    for i in range(len(samples_sampler)*skipIter, i_iter+1, skipIter):
        sampler_sampler = ais.get_sampler(i)
        a = np.array([sampler_sampler.get_sample() for _ in range(nSamples)])
        samples_sampler = np.append(samples_sampler, a.reshape(1,a.shape[0], a.shape[1]), axis=0)

    np.save(filename, samples_sampler, allow_pickle=True)

    samples_sampler = np.array(samples_sampler)

    if ais.patient.modelPdict!=[]:
        thetaRef = ais.parameters.getX(ais.model.getInstance(Pdict=ais.patient.modelPdict))

    


    plt.figure()
    for iPar in range(len(ais.parameters)):
        print('plot ', iPar, ' of 19')
        plt.subplot(4,5,iPar+1)

        samples_sampler

        s = samples_sampler[:,:,iPar]

        sHist = []

        for i_iter in range(s.shape[0]):
            hist, b = np.histogram(s[i_iter,:], bins=1000, range=[np.min(s), np.max(s)])
            sHist.append(np.array(hist))

        sHist = np.array(sHist).transpose()

        #maxHist = np.max(sHist)
        itermax = np.max(sHist, axis=0)


        # change color
        #idx = itermax<maxHist*0.2
        #sHist[:,idx] = sHist[:,idx] / itermax[idx] * 0.2 * maxHist
        #sHist[sHist>500]=500

        sHist = sHist / itermax**0.1

        sHist =sHist ** 0.4



        xv, yv = np.meshgrid(b[:-1], range(s.shape[0]), sparse=False, indexing='ij')
        plt.contourf(xv, yv, sHist, extend='both', cmap = mpl.cm.binary)
        plt.gca().invert_yaxis()

        plt.plot(np.min(s, axis=1), range(s.shape[0]), c=[0,0,0])
        plt.plot(np.max(s, axis=1), range(s.shape[0]), c=[0,0,0])

        try:
            plt.scatter(thetaRef[iPar], s.shape[0]-1, ec=[0,0,0], fc=[1,1,1], s=100)
        except:
            print('no reference values available')

        plt.xlabel(ais.parameters.thetaGetLable(iPar=iPar,includeLocation=True))
        plt.title(ais.parameters.thetaGetTitle(iPar=iPar))
        
        ###### kernel
        
