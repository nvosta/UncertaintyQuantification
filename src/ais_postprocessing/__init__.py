import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

def print_ais(ais_list):
    for ais in ais_list:
        print('------------')
        print('Patient: ', ais.patient.id)
        print('min X2: ', np.min(ais.prior_data['theta_info']['X2']))
        print('number of sims: ', len(ais.prior_data['theta_info']['X2']))
        print('number of sims: ', len(ais.prior_data['theta_info']['X2']))
        X2 =  ais.prior_data['theta_info']['X2']
        q =  ais.prior_data_merge_q(ais.prior_data)
        w = np.exp(-X2-q)
        n_eff = np.sum(w)**2 / np.sum(w**2)
        print('number of effective sims: ', n_eff)
            

def plot_1d(ais_list, data='theta'):
    plt.clf()
    plt.subplots_adjust(top=0.95,
        bottom=0.05,
        left=0.11,
        right=0.9,
        hspace=0.5,
        wspace=0.2)
    
    # ais_list should contain similar results
    parameters = ais_list[0].parameters
    
    m,n = 4,5
    if len(parameters)<5:
        m,n=1,len(parameters)
    elif len(parameters)<=20:
        m,n=4,5
    elif len(parameters)<=25:
        m,n=5,5
    

    print('Get Dummy samples')
    for ais in ais_list:
        theta = ais.prior_data['theta_info']['theta']
        X2 = ais.prior_data['theta_info']['X2']
        q = ais.prior_data_merge_q(ais.prior_data)
        
    theta = []
    thetaRef = []
    for i_ais in range(len(ais_list)):
        if data=='theta':
            theta.append(ais_list[i_ais].prior_data['theta_info']['theta'])
        else:
            theta.append(ais_list[i_ais].parameters.XtoParVal( np.array(ais_list[i_ais].prior_data['theta_info']['theta'])))
            
        # thera ref
        if ais_list[i_ais].patient.modelPdict != []:
            model_instance = ais_list[i_ais].model.getInstance()
            model_instance.setPdict(ais_list[i_ais].patient.modelPdict)
            
            tr = ais_list[i_ais].parameters.getX(model_instance)
            if data=='theta':
                thetaRef.append(tr)
            else:
                thetaRef.append(ais_list[i_ais].parameters.XtoParVal( np.array(tr)))
                
            
            
    
    
    
    colors = [[1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0.5], [0.2,0.5,0.7], [0.7,0.5,0.2]]
    ax=[[] for _ in range(len(theta[0][0]))]
    for i_par in range(len(theta[0][0])):
        print('Plot par ', i_par)
        ax[i_par] = plt.subplot(m,n,i_par+1)
        #plt.hist(dummy_samples[0][:,i_par])
        
        # Kolmogorov Smirnov test
        theta_range = np.array([np.Infinity, -np.Infinity])
        for i_ais in range(len(ais_list)):
            theta_range[0] = np.min([theta_range[0], np.min(theta[i_ais][:,i_par])])
            theta_range[1] = np.max([theta_range[1], np.max(theta[i_ais][:,i_par])])
        prob = np.ndarray((len(ais_list), 500))
        cumProb = np.ndarray((len(ais_list), 500))
        n_eff = np.zeros(len(ais_list))
        
        for i_ais in range(len(ais_list)):
            theta1 = theta[i_ais][:,i_par]
            X2 =  ais_list[i_ais].prior_data['theta_info']['X2']
            q =  ais_list[i_ais].prior_data_merge_q(ais_list[i_ais].prior_data)
            
            #if data=='parameters' and i_par not in [10,11,12,13,14]: 
            #    q = q / X2 # log transformation, not for dT
            
            w = np.exp(-X2-q)
            
            idx = (w > 1e-10*np.max(w) ) & (X2 < np.min(X2)+10)
            
            n_eff[i_ais] = np.sum(w)**2 / np.sum(w**2)
            
            hi, bi = np.histogram(theta1, bins=int(n_eff[i_ais]), weights=w)
            #plt.bar(bi[:-1], hi)
            #plt.ylim([0, np.max(hi)*1.1])
            
            # get bandwidth
            cshi = np.append(0,np.cumsum(hi))
            l = bi[cshi>=0.25*cshi[-1]][0]
            r = bi[cshi>=0.75*cshi[-1]][0]
            bandwidth = r-l + bi[1] - bi[0]
            bandwidth = bandwidth / 1.3 * len(bi)**(-1/4)
            
            kde = KernelDensity(
                bandwidth=bandwidth, kernel='gaussian')
            idx = hi>1e-3*np.max(hi)
            theta_fit = (bi[:-1]+0.5*np.diff(bi))[idx].reshape(-1,1)
            x_d = np.linspace(theta_range[0], theta_range[1], cumProb.shape[1])
            
            kde.fit(theta_fit, sample_weight=hi[idx])
            # score_samples returns the log of the probability density
            logprob = kde.score_samples(x_d[:, None])
            
            
            prob[i_ais, :] = np.exp(logprob) / (np.sum(np.exp(logprob)) * (x_d[1]-x_d[0]))
            cumProb[i_ais, :] = np.cumsum(prob[i_ais, :])
            cumProb[i_ais, :] = cumProb[i_ais, :] #/ cumProb[i_ais, -1] 
            
            plt.fill_between(x_d, prob[i_ais, :], alpha=0.25, fc=colors[i_ais])
            plt.plot(x_d, prob[i_ais, :], c=colors[i_ais])
            
            #plt.fill_between(x_d, cumProb[i_ais, :], alpha=0.25, fc=colors[i_ais])
            #plt.plot(x_d, cumProb[i_ais, :], c=colors[i_ais])
            
            dd = x_d[1]-x_d[0]
            domain_of_interest = x_d[prob[i_ais, :]>np.max(prob[i_ais, :])*0.01]
            plt.xlim([np.min(domain_of_interest), np.max(domain_of_interest)])
            
            
            
            # theta ref
            if len(thetaRef)>i_ais:
                plt.scatter(thetaRef[i_ais][i_par], 0, ec='k',fc=[1,1,1])
        
        # Kolmogorov Smirnov test
        D = np.zeros((len(ais_list), len(ais_list)))
        A = np.zeros((len(ais_list), len(ais_list)))
        for i_ais0 in range(len(ais_list)):
            for i_ais1 in range(len(ais_list)):
                D[i_ais0, i_ais1] = np.max(np.abs(cumProb[i_ais0,:]-cumProb[i_ais1,:]))
                if False:#i_ais0 == i_ais1:
                    A[i_ais0, i_ais1] = np.sum(prob[i_ais0,:]) * (x_d[1]-x_d[0])
                else:
                    A[i_ais0, i_ais1] = np.sum(np.min(np.array([prob[i_ais0,:],prob[i_ais1,:]]), axis=0)) * (x_d[1]-x_d[0])
        
        alpha = 0.0001
        KSn = n_eff[np.array([range(len(ais_list)),]*len(ais_list))]
        KSm = n_eff[np.array([range(len(ais_list)),]*len(ais_list))].transpose()
        KSconst =  np.sqrt(-np.log(alpha/2)*0.5 * (KSn+KSm)/(KSn*KSm))
        print('Kolmogorov Smirnov test i_par ', i_par, '. The follow distributions are the same: ')
        print(D < KSconst)
        print(A)
        distributions_are_similar = np.any((D < KSconst)==False)==False
        
                
        if data=='theta':
            KStitle = ''
            if distributions_are_similar:
                KStitle = '*'
                
            try:
                plt.title(ais_list[i_ais].parameters.thetaGetTitle(i_par) + KStitle)
            except:
                plt.title(KStitle)
            
            try:
                plt.xlabel(ais_list[i_ais].parameters.thetaGetLable(i_par))
            except:
                pass
            plt.ylabel(r'$p(\theta|z)$')
        elif data=='parameters':
            plt.xlabel(ais_list[i_ais].parameters.parameterGetLabelModel(iPar=i_par,includeLocation=True))
            plt.title(ais_list[i_ais].parameters.parameterGetLabelClinical(iPar=i_par))
            
            
        #plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
        #plt.ylim(-0.02, 0.22)
        
    if data=='parameters': 
        same_axis=[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17]]
        for sa in same_axis:
            xl=[np.inf, -np.inf]
            for isa in sa:
                axl = ax[isa].get_xlim()
                xl[0] = np.min([xl[0], axl[0]])
                xl[1] = np.max([xl[1], axl[1]])
            for isa in sa:
                ax[isa].set_xlim(xl)
            
            
        
    

def scatter_2d(ais, data='X2'):
    plt.clf()
    plt.subplots_adjust(bottom=0.01, right=0.9, top=0.99)
    
    n_par = len(ais.parameters)
    
    for i_par_0 in range(n_par):
        for i_par_1 in range(i_par_0, n_par):
            if i_par_0==i_par_1:
                pass
            else:
                plt.subplot(n_par, n_par, i_par_0*n_par+i_par_1+1)
                theta = ais.prior_data['theta_info']['theta']
                X2 = ais.prior_data['theta_info']['X2']
                plt.scatter(theta[:,i_par_0], theta[:,i_par_1], c=X2)
                if i_par_0+1==i_par_1:
                    plt.ylabel(r'$\theta_'+str(i_par_1)+'$')
                    plt.xlabel(r'$\theta_'+str(i_par_0)+'$')
                
                cbar= plt.colorbar()

    return 0
    
    #%% post processing
    theta = ais.prior_data['theta_info']['theta']
    X2 = ais.prior_data['theta_info']['X2']
    q = ais.prior_data_merge_q(ais.prior_data)
    k = ais.get_sampler(len(ais.prior_data['iter_info']['i_iter']), ais.prior_data)
    
    plt.figure(1)
    plt.clf()
    
    plt.subplot(3,2,1)
    plt.title('Train data X2')
    plt.scatter(theta[:,0], theta[:,1], c=X2)
    cbar= plt.colorbar()
    
    plt.subplot(3,2,2)
    plt.title('Train data X2')
    plt.scatter(theta[:,0], 
                theta[:,1], 
                c=k.score_gamma_weight(
                    theta, 
                    X2, 
                    q)[1]
                )
    cbar= plt.colorbar()
    xl = plt.xlim()
    yl = plt.ylim()
    
    
    plt.subplot(3,2,4)
    theta = k.get_samples(1000)
    score = np.exp(k.score_samples(theta))
    plt.scatter(theta[:,0], theta[:,1], c=score)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    cbar= plt.colorbar()
    plt.title('Result from sampler')
    plt.xlim(xl)
    plt.ylim(yl)
    
    for i_p in [0,1]:
        plt.subplot(3,2,5+i_p)
        plt.hist(theta[:,i_p])
        plt.xlabel(r'$\theta_'+str(i_p)+'$')
        
        
        
def plot_strain_best_X2(ais, i_best=0):
    i_theta = np.argsort(ais.prior_data['theta_info']['X2'])[i_best]
    theta = ais.prior_data['theta_info']['theta'][i_theta,:]
    patient = ais.patient
    
    model_instance = ais.model.getInstance()
    theta0 = ais.parameters.getX(model_instance)
    model_instance.setScalar('','','','tCycle', round(patient.RVtime[-1]/0.002)*0.002)
    

    for alpha in np.linspace(0.1,1,10):
        ais.parameters.setX(model_instance, alpha*theta+(1-alpha)*theta0)
        model_instance.run()
    ais.parameters.setX(model_instance, theta)
    model_instance.run()
    
    
    ais.cost.plotValidation(patient=patient, model_instance=model_instance)
    
    
    plt.figure()
    plt.subplot(2,2,1)
    t = model_instance.getTime()
    plt.plot(t, np.array(model_instance.getVector('Lv','Cavity','','p'))*0.00750061683 )
    plt.plot(t, np.array(model_instance.getVector('Rv','Cavity','','p'))*0.00750061683 )
    plt.plot(t, np.array(model_instance.getVector('Ra','Cavity','','p'))*0.00750061683 )
    plt.plot(t, np.array(model_instance.getVector('La','Cavity','','p'))*0.00750061683 )
    
    plt.plot([t[0], t[-1]], [10 , 10 ],'--',c=[.5,.5,.5])
    plt.plot([t[0], t[-1]], [15 , 15 ],'--',c=[.5,.5,.5])
    
    plt.subplot(2,2,2)
    t = model_instance.getTime()
    plt.plot(t, np.array(model_instance.getVector('Lv','Cavity','','V'))*1e6)
    plt.plot(t, np.array(model_instance.getVector('Rv','Cavity','','V'))*1e6)
    #plt.plot(t, np.array(model_instance.getVector('Ra','Cavity','','V')))
    #plt.plot(t, np.array(model_instance.getVector('La','Cavity','','V')))
    
    plt.subplot(2,2,3)
    t = model_instance.getTime()
    plt.plot(t, model_instance.getVector('Lv','Wall','','Cm'))
    plt.plot(t, model_instance.getVector('Sv','Wall','','Cm'))
    plt.plot(t, model_instance.getVector('Rv','Wall','','Cm'))
    #plt.plot(t, model_instance.getVector('Ra','Cavity','','V'))
    #plt.plot(t, model_instance.getVector('La','Cavity','','V'))
    
    
    ax = plt.subplot(2,2,4)
    t = model_instance.getTime()
    
    Pdict = model_instance.getPdict()
    
    Am = Pdict['Wall']['Am'][:,6:9]
    Ym = Pdict['TriSeg']['Y']
    signCm = np.sign(Pdict['Wall']['Cm'][:,6:9])
    
    Xm = signCm * np.sqrt(Am/np.pi - Ym**2)
    ax.plot(Xm)
    
    ax = plt.twinx()
    ax.plot( (Xm[:,2]-Xm[:,1]) / (Xm[:,1]-Xm[:,0]))
    
    print('RV EF: ', 1-np.min(model_instance.getVector('Rv','Cavity','','V'))/np.max(model_instance.getVector('Rv','Cavity','','V')))
    
    return model_instance
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def plot_temperature_X2_over_time(ais):
    plt.clf()
    plt.subplot(3,1,1)
    plt.plot(ais.prior_data['iter_info']['max_q'])
    plt.xlabel('iteration')
    plt.ylabel('max_q')
    
    plt.subplot(3,1,2)
    plt.plot(ais.prior_data['iter_info']['X2min'])
    plt.xlabel('iteration')
    plt.ylabel('min X2')
    plt.yscale('log')
    
    plt.subplot(3,1,3)
    max_temp = ais.prior_data['iter_info']['max_temperature']
    # correct
    max_temp[1:] = max_temp[:-1]
    #max_temp[:-1] = max_temp[1:]
    #max_temp[0]=np.inf
    temp = ais.prior_data['iter_info']['temperature']
    max_temp = np.max(np.array([max_temp, temp]), axis=0)
    plt.plot(temp)
    plt.plot(max_temp)
    plt.xlabel('iteration')
    plt.ylabel('Temperature')
    plt.legend({'Actual temperature', 'Maximum temperature'})
    
    