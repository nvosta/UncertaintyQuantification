"""
    Kernels

    by Nick van Osta
"""

import numpy as np

from .kernel1 import kernel1
from .kernel1 import kernel1a
from .kernel1 import kernel1b
from .kernel1 import kernel1c

class kernel:
    """
    Kernel.

    used for building proposal distribution
    """

    #n_par = 1
    #W = np.eye(n_par)
    #M = np.zeros(n_par)
    #K = []
    #w_k = []
    #T = np.Infinity

    def __init__(self, theta=[], X2=[], pi=[], n_par = None, Tmin = 1, Tmax = np.Infinity, sim_fac = 1, new_sim_damper = True, fac_width=-1, logcorrection=None):
        #if fac_width < 0:
        #    fac_width=1 if len(X2)<1000 else 0
            
        if not n_par is None:
            self.n_par = n_par
        else:
            self.n_par = theta.shape[1]

        self.Tmin = Tmin
        self.Tmax = Tmax
        self.T = Tmax
        self.Ts = np.Inf
        self.sim_fac = sim_fac

        self.W = np.eye(self.n_par)
        self.Winv = np.eye(self.n_par)
        self.M = np.zeros(self.n_par)
        
        self.n_kernel = 3

        self.K = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_k = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_s = [1.0,0.0,0.0]# w, w1, ws
        self.w_k1 = np.array(self.w_s, dtype=np.double )
        
        self.new_sim_damper = new_sim_damper

        if len(theta)>0:
            self.fit(np.array(theta), np.array(X2), np.array(pi), fac_width=fac_width)

    def score_gamma_weight(self,theta, X2, pi, Tmin, Tmax, logcorrection=None):
        # Determine temperature
        T = Tmin
        
        if len(X2)< (self.n_par * self.sim_fac*2):
            T = T*100
        
        if logcorrection is None:
            logcorrection = np.zeros(X2.shape)

        ws = np.exp( (-X2-pi+logcorrection) / T -logcorrection)
        if False:
            while (np.sum(ws)<1e-100 or get_effective_number_of_samples(ws) < self.n_par * self.sim_fac) and T < Tmax:
                if T>=100:
                    T=T*10 
                else:
                    T=T+np.log10(T+0.01) # only small steps when T is low
                if T>Tmax or len(ws)<self.n_par * self.sim_fac:
                    T=np.max([Tmin, Tmax])
                    ws = np.zeros(len(ws))
                    ws[np.argsort(X2)[:int(self.n_par * self.sim_fac)]]=1
                    break
                ws = np.exp( (-X2-pi+logcorrection) / T - logcorrection)
        else:
            # newton raphson
            
            n_eff_target = self.n_par * self.sim_fac
            Ttrymin = 1
            Ttrymax = 100
            for i in range(5):
                
                
                Ttry = np.linspace(Ttrymin, Ttrymax, 10)
                n_eff_try = np.array([get_effective_number_of_samples(np.exp( (-X2-pi+logcorrection) / Ttry[i] - logcorrection)) for i in range(len(Ttry))])
                
                if n_eff_try[0]>n_eff_target:
                    return 1, ws
                elif n_eff_try[-1]<n_eff_target:
                    Ttrymax = Ttrymax * 10
                else:
                    Ttrymin = Ttry[np.argmax(n_eff_try>n_eff_target)-1]
                    Ttrymax = Ttry[np.argmax(n_eff_try>n_eff_target)]
                
            if n_eff_try[-1]<n_eff_target:
                T = Ttrymax
            else:
                T = np.interp(n_eff_target, n_eff_try, Ttry)
            ws = np.exp( (-X2-pi+logcorrection) / T - logcorrection)
            n_eff = get_effective_number_of_samples(ws)
                
                
            
            
        #print('Temperature: ', T)
        return T, ws

    def getBuildWAndWw(self, theta, X2, pi, T, w, ws):
        w1 = np.exp(-X2/(0.5*T)-pi)
        print('Effective samples w1: ', get_effective_number_of_samples(w1))
        
        # Build sample w
        buildW = [w, w1, ws]
        buildWw = self.w_k1 
        
        return (buildW, buildWw)

    def fit(self, theta, X2, pi, fac_width=1, logcorrection=None):
        """Fit."""
        #T, ws = self.score_gamma_weight(theta, X2, pi/self.Tmin, np.max([self.Tmin, 1]), 1000000*self.Tmax, logcorrection=logcorrection)
        T, ws = self.score_gamma_weight(theta, X2, pi, 1, 1000000*np.max([10,self.Tmax]), logcorrection=logcorrection)
        print('Temperature ws: ', T)
        self.Ts = T
        T = np.min([T, self.Tmax])
        print('Temperature w: ', T)
        print('Effective samples ws: ', get_effective_number_of_samples(ws))
        self.T = T

        # get sample weight
        w = np.exp(-X2/T-pi)
        print('Effective samples w : ', get_effective_number_of_samples(w))
        # lower the weight of most recent samples to prevent from collapsing
        if False:#self.new_sim_damper:
            n_sims = 100
            fac_sim = np.min([4, 0.5*len(w)/n_sims])
            if fac_sim>=1:
                max_w = np.max(w[:-int(fac_sim*n_sims)])
                w[-1*int(fac_sim*n_sims):][w[-1*int(fac_sim*n_sims):]>max_w] = max_w
        
        print('Min X2: ', np.min(X2))
        

        # determine principle components
        # use w to rotate around current best optimum
        self.calc_M_and_W(theta, w, ws)

        # transform
        thetas = np.ndarray(theta.shape)
        for iTheta in range(theta.shape[0]):
            thetas[iTheta,:] = np.dot(self.Winv,theta[iTheta,:]-self.M)
            
        # get sample weight
        buildW, buildWw = self.getBuildWAndWw(thetas, X2, T, pi, w, ws)
            
        # get n_bins
        # freedman - diaconis rule, nBins = (max-min)/h, h=2 * IQR * n^(-1/3)
        n_bins = []
        
        
        # apply gamma for ws for bins so bins are not based on a single sample
        # otherwise, when n_eff=1, nbins->inf
        wsbin = np.array(ws) / np.sum(ws)
        while get_effective_number_of_samples(wsbin) < self.n_par:
            wsbin = wsbin**(1/2)
            wsbin = wsbin / np.sum(wsbin)
        
        # define number of bins
        IQR = []
        for i_par in range(self.n_par):# determine nbins for this theta
            asTheta = np.argsort(thetas[:,i_par])
            cssw = np.cumsum(wsbin[asTheta])

            t0 = np.min(thetas[:,i_par])
            t05 = thetas[ asTheta[len(cssw)-(cssw<0.05*cssw[-1])[::-1].argmax() - 1], i_par ]
            t25 = thetas[ asTheta[len(cssw)-(cssw<0.25*cssw[-1])[::-1].argmax() - 1], i_par ]
            t75 = thetas[ asTheta[(cssw>0.75*cssw[-1])[::1].argmax()] , i_par]
            t95 = thetas[ asTheta[(cssw>0.95*cssw[-1])[::1].argmax()] , i_par]
            t100 = np.max(thetas[:,i_par])

            # use max to prevent from collapsing
            IQR.append(t75-t25)
            n_bins.append(
                                   (t100-t0) / 
                                   (np.max([0.01*(t100-t0), (t75-t25)]) ) 
                          )
            
            
        thetasOpt = np.sum(thetas*w.reshape(-1,1)/np.sum(w), axis=0)

        # loop over two sample weights
        #for sample_weight, kernel_weight in zip(buildW, buildWw):
        for iW in range(len(buildW)):
            sample_weight = buildW[iW]

            # set maximum
            #n_eff = get_effective_number_of_samples(sample_weight) # n_eff of this set
            n_eff = get_effective_number_of_samples(w) # n_eff of total sapmle set
            ploMax = 1#1-1/np.max([1.00001, n_eff])
            ploMin = 0.01
            
            #self.w_k1[iW] = self.w_s[iW] #* np.max([ploMax, ploMin])

            if ploMax < ploMin:
                plo = np.array([ploMin])
            else:
                plo = np.linspace(ploMin, ploMax, 10)

            w_plo = np.diff(np.append(0, plo))
            binFac = 2*n_eff**(-1/3)
            for i_par in range(self.n_par):

                bins = int(n_bins[i_par] / binFac) 
                bins = np.max([bins, int(np.max(thetas[:,i_par])-np.min(thetas[:,i_par]))])
                hi, bi = np.histogram(thetas[:,i_par], 
                                      weights=sample_weight, 
                                      bins=bins, 
                                      range=(np.min(thetas[:,i_par]), np.max(thetas[:,i_par])))

                binwidth = bi[1] - bi[0]

                for i_plo in range(len(plo)):
                    # get min and max of histogram above plo[i_plo]
                    thetaMin, _, thetaMax, height = histFindMinOptMax(hi, bi, plo[i_plo])
                    thetaOpt = thetasOpt[i_par]

                    # add width to kernel to prevent from collaps
                    #thetaMin = np.min([thetaMin-binwidth, thetaOpt-0.1*IQR])
                    #thetaMax = np.max([thetaMax+binwidth, thetaOpt+0.1*IQR])
                    
                    thetaMin = np.min([thetaMin-0.05*binwidth, thetaOpt-0.1*IQR[i_par]])
                    thetaMax = np.max([thetaMax+0.05*binwidth, thetaOpt+0.1*IQR[i_par]])
                    
                    
                    # create kernel
                    k = tophat(
                        thetaMin,
                        thetaMax
                    )

                    # check if kernel exists
                    addKernel = True
                    for i_k in range(len(self.K[iW][i_par])):
                        if self.K[iW][i_par][i_k].a == k.a and self.K[iW][i_par][i_k].b == k.b:
                            self.w_k[iW][i_par][i_k]+=w_plo[i_plo] 
                            addKernel = False
                            break
                    if addKernel:
                        self.K[iW][i_par].append(k)
                        self.w_k[iW][i_par].append( w_plo[i_plo] )







    def add_tophat(self, i_par, a, b, w_k, iW=0):
        """Add a tophat kernel to the kernels."""
        self.K[iW][i_par].append(tophat(a, b))
        self.w_k[iW][i_par].append(w_k)

    def get_samples(self, n_samples):
        """Get multiple samples."""
        return np.array([self.get_sample() for _ in range(n_samples)])

    def score_samples(self,theta):
        """Score."""
        # backward compatibility, remove line in future
        self.Winv = np.linalg.inv(self.W)
        
        X = np.array(theta)
        for iTheta in range(X.shape[0]):
            X[iTheta,:] = np.dot(self.Winv,X[iTheta,:]-self.M)

        scores = np.zeros((X.shape[0], X.shape[1]))
        
        for iW in range(len(self.w_k1)):
            for i_par in range(scores.shape[1]):
                for i_k in range(len(self.w_k[iW][i_par])):
                    s = self.K[iW][i_par][i_k].score_samples(X[:,i_par])
                    scores[:, i_par]+= np.exp(s) * self.w_k[iW][i_par][i_k] / np.sum(self.w_k[iW][i_par]) * self.w_k1[iW] / np.sum(self.w_k1)



        # combine scores
        idx = scores<1e-100
        scores[idx] = -np.Infinity
        scores_transform_correction = -np.log(np.abs(np.linalg.det(self.W)))
        #scores[idx==False] = np.log(scores[idx==False]) + self.max_single_weight + scores_transform_correction
        scores[idx==False] = np.log(scores[idx==False]) + scores_transform_correction

        scores = np.sum(scores, axis=1)

        return scores


    def get_sample(self):
        """Get a sample"""
        sample = np.ndarray(self.n_par)
        
        # select iW
        u = np.random.random()
        cumsum_weight = np.cumsum(self.w_k1)
        sum_weight = cumsum_weight[-1]
        iW = np.searchsorted(cumsum_weight, u * sum_weight)
        
        
        for i_par in range(len(sample)):
            u = np.random.random()
            cumsum_weight = np.cumsum(self.w_k[iW][i_par])
            sum_weight = cumsum_weight[-1]
            i = np.searchsorted(cumsum_weight, u * sum_weight)

            # get sample
            sample[i_par] = self.K[iW][i_par][i].get_sample()

        return np.dot(self.W, sample) + self.M
    
    def calc_M_and_W(self, theta, w1, w2):
        self.M = np.average(theta,axis=0, weights=w1) 
        if theta.shape[0]>theta.shape[1]:
            eigvec = np.linalg.eig( np.cov( (theta-self.M).transpose(), aweights = w2 )  )
            self.W = eigvec[1]
        else:
            self.W = np.eye(theta.shape[1])
        self.Winv = np.linalg.inv(self.W)
    
#%% Kernel density for final
class kernelDensity (kernel):
    def __init__(self, theta=[], X2=[], pi=[]):
        
        self.n_kernel = theta.shape[0]
        self.n_par = theta.shape[1]
        
        self.K = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_k = [[[1] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_s = []# w, w1, ws
        self.w_k1 = []
        
        self.fit(theta, X2, pi)
    
    def fit(self, theta, X2, pi):
        w = np.exp(-X2-pi)
        n_eff = get_effective_number_of_samples(w)
        self.w_k1 = w
        
        # transform
        self.calc_M_and_W(theta, w, w)
        thetas = np.ndarray(theta.shape)
        for iTheta in range(theta.shape[0]):
            thetas[iTheta,:] = np.dot(self.Winv,theta[iTheta,:]-self.M)
        
        # bandwidth
        
        band_width = []
        for i_par in range(theta.shape[1]):
            asTheta = np.argsort(thetas[:,i_par])
            cssw = np.cumsum(w[asTheta])
            
            t25 = thetas[ asTheta[len(cssw)-(cssw<0.25*cssw[-1])[::-1].argmax() - 1], i_par ]
            t75 = thetas[ asTheta[(cssw>0.75*cssw[-1])[::1].argmax()] , i_par]
            IQR = t75 - t25
            
            band_width.append(0.1*0.9*n_eff**(-1/5)*IQR/1.34)
            
        # create        
        for i_sample in range(theta.shape[0]):
            for i_par in range(theta.shape[1]):
                thetaMin = thetas[i_sample][i_par] - 0.05*band_width[i_par]
                thetaMax = thetas[i_sample][i_par] + 0.05*band_width[i_par]
                self.K[i_sample][i_par].append(tophat(
                        thetaMin,
                        thetaMax
                    ))
                
                
#%% Kernel Small
class kernelSmall (kernel):                
    def getBuildWAndWw(self, theta, X2, pi, T, w, ws):
        
        nKernels = 100
        
        w1 = [np.array(w)]
        
        #np.zeros(w.shape)

        
        
        IQR = np.ndarray(theta.shape[1])
        for i_par in range(theta.shape[1]):
            asTheta = np.argsort(theta[:,i_par])
            cssw = np.cumsum(ws[asTheta])
            
            t05 = theta[ asTheta[len(cssw)-(cssw<0.05*cssw[-1])[::-1].argmax() - 1], i_par ]
            t95 = theta[ asTheta[(cssw>0.95*cssw[-1])[::1].argmax()] , i_par]
            IQR[i_par] = t95 - t05
            
        w1Tot = np.sum(w1[0])
            
        while len(w1)<nKernels:
            
            # do not add kernels if weight is neglectible
            if np.sum(w1[0]) < 0.01*w1Tot:
                break
            
            w1n = np.zeros(w.shape)
            idx_w_max = np.argmax(w1[0])
            
            theta_max = theta[idx_w_max,:]
            
            idx_include_theta = ((theta>(theta_max-IQR*0.5))&(theta<(theta_max+IQR*0.5)))
            idx_include_theta = np.any(idx_include_theta==False, axis=1)==False
            
            w1n[idx_include_theta] = w1[0][idx_include_theta]
            w1[0][idx_include_theta] = 0
            w1.append(w1n)
            

        # Build sample w
        buildW = w1
        self.w_k1 = [np.sum(w1[i]) for i in range(len(w1))]
        
        self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
        idxLow = self.w_k1>0.5
        self.w_k1[idxLow] = 0.5 
        self.w_k1[idxLow==False] = self.w_k1[idxLow==False] / np.sum(self.w_k1[idxLow==False]) * (1-np.sum(self.w_k1[idxLow]))
        self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
        
        
        self.n_kernel = len(w1)
        self.K = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_k = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        
        buildWw = self.w_k1 
        print('number of kernels: ',len(self.w_k1 ))
        
        return (buildW, buildWw)
    
    
class kernelKMeans(kernel):                
    def getBuildWAndWw(self, theta, X2, pi, T, w, ws):
        
        nKernels = np.max([2, int(np.min([get_effective_number_of_samples(w)/2, 100])) ])
        
        w1 = [np.array(w)]

        
        # Train KMEans
        from sklearn.cluster import KMeans  #For applying KMeans
        kmeans = KMeans(n_clusters=nKernels, n_init=10, random_state=0, max_iter=1000)

        idx = w>0.01
        if np.sum(idx)<nKernels*2:
            idx = np.argsort(w)[::-1][:nKernels*2]

        kmeans.fit(theta[idx,:], sample_weight=w[idx])
        idx = kmeans.predict(theta)
        
        w1 = np.zeros(( nKernels, len(w)))
        for i in range(nKernels):
            idx1 = idx==i
            w1[i,idx1] = w[idx1]

        # Build sample w
        buildW = w1
        self.w_k1 = [np.sum(w1[i]) for i in range(len(w1))]
        
        self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
        idxLow = self.w_k1<1/(len(self.w_k1)*1000)
        self.w_k1[idxLow] = 1/(len(self.w_k1)*1000)
        
        idxLow = self.w_k1>0.5
        self.w_k1[idxLow] = 0.5 
        self.w_k1[idxLow==False] = self.w_k1[idxLow==False] / np.sum(self.w_k1[idxLow==False]) * (1-np.sum(self.w_k1[idxLow]))
        self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
        
        self.n_kernel = len(w1)
        self.K = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_k = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        
        buildWw = self.w_k1 
        print('number of kernels: ',len(self.w_k1 ))
        
        return (buildW, buildWw)
    
class kernelKMeans1(kernel):                
    def getBuildWAndWw(self, theta, X2, pi, T, w1, ws):
        
        w = np.array(w1)
        if len(w)>200:
            maxW = np.max(w[:-200])
            w[w>maxW]=maxW
        
        
        nKernels = np.max([2, int(np.min([get_effective_number_of_samples(w)/2, 100])) ])
        
        w1 = [np.array(w)]

        
        # Train KMEans
        from sklearn.cluster import KMeans  #For applying KMeans
        kmeans = KMeans(n_clusters=nKernels, n_init=10, random_state=0, max_iter=1000)

        idx = w>0.01
        if np.sum(idx)<nKernels*2:
            idx = np.argsort(w)[::-1][:nKernels*2]

        kmeans.fit(theta[idx,:], sample_weight=w[idx])
        idx = kmeans.predict(theta)
        
        w1 = np.zeros(( nKernels, len(w)))
        for i in range(nKernels):
            idx1 = idx==i
            w1[i,idx1] = w[idx1]

        # Build sample w
        buildW = w1
        self.w_k1 = [np.sum(w1[i]) for i in range(len(w1))]
        
        self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
        idxLow = self.w_k1<1/(len(self.w_k1)*1000)
        self.w_k1[idxLow] = 1/(len(self.w_k1)*1000)
        
        idxLow = self.w_k1>0.5
        self.w_k1[idxLow] = 0.5 
        self.w_k1[idxLow==False] = self.w_k1[idxLow==False] / np.sum(self.w_k1[idxLow==False]) * (1-np.sum(self.w_k1[idxLow]))
        self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
        
        self.n_kernel = len(w1)
        self.K = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_k = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        
        buildWw = self.w_k1 
        print('number of kernels: ',len(self.w_k1 ))
        
        return (buildW, buildWw)

#%% Essentials
def get_effective_number_of_samples(w):
    if np.sum(w**2)<1e-300:
        return 1
    return np.sum(w)**2 / np.sum(w**2)


def histFindMinOptMax(hi, bi, fac):
    sum_hist = np.sum(hi)
    if sum_hist<1e-100:
        return np.min(bi), np.mean(bi), np.max(bi), 0

    hist_tmp = np.array(hi)

    fac_act = 0
    while fac_act < fac:
        min_hist = np.min(hist_tmp[hist_tmp>0])

        if fac_act + min_hist * np.sum(hist_tmp>0) / sum_hist < fac:
            hist_tmp[hist_tmp>0] = hist_tmp[hist_tmp>0] - min_hist
            fac_act = 1-np.sum(hist_tmp) / np.sum(hi)
        else:
            d = (np.sum(hist_tmp) - sum_hist*(1-fac)) / np.sum(hist_tmp>0)
            hist_tmp[hist_tmp>0] = hist_tmp[hist_tmp>0] - d
            fac_act = 1-np.sum(hist_tmp) / np.sum(hi)
            break

    arghist = np.argwhere(hist_tmp>0)
    if len(arghist) ==0:
        return histFindMinOptMax(hi, bi, fac*0.99)
    thetaMin = bi[arghist[0,0]]
    thetaMax = bi[arghist[-1,-1]+1]
    thetaOpt = (bi[np.argmax(hist_tmp)]+bi[np.argmax(hist_tmp)+1])/2

    # calculate hight from 0
    h = 1 - np.max(hist_tmp) / np.max(hi)

    return thetaMin, thetaOpt, thetaMax, h



## 1D kernels
class tophat:
    """
    Kernel.

    used for building proposal distribution
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.s = -np.log(self.b-self.a)

    def get_sample(self):
        """Get a sample."""
        u = np.random.random()
        return u * (self.b - self.a) + self.a

    def score_samples(self, theta):
        """Score."""
        s = np.ones(len(theta))*self.s
        s[theta<self.a]=-np.Infinity
        s[theta>self.b]=-np.Infinity
        return s

class final_sampler:
    def __init__(self, ais):
        pass
    

        