# -*- coding: utf-8 -*-
import numpy as np


class kernel1:
    def __init__(self, theta=[], X2=[], pi=[], Tmin=1, Tmax=np.Inf, logcorrection=None, useKMeans=True):
        
        if logcorrection is None:
            logcorrection = np.zeros(X2.shape)
        
        self.n_kernel = 1
        self.n_par = theta.shape[1]
        self.n_bins = 200
        self.sim_fac = 10
        self.Tmin = Tmin
        self.Tmax = Tmax
        
        self.fit(np.array(theta), np.array(X2), np.array(pi), np.array(logcorrection))
        
    def fit(self, theta, X2, pi, logcorrection):
        Tg, wg = self.score_gamma_weight(theta, X2, pi, Tmin=self.Tmin, logcorrection=logcorrection)
        T = np.max([np.min([Tg, self.Tmax]), self.Tmin])
        self.T = T
        self.Ts = Tg
        w = np.exp(-X2/T-pi)
        N_eff = get_effective_number_of_samples(w)
        N_effg = get_effective_number_of_samples(wg)
        
        print('#-----------')
        print('# Fit kernel ')
        print('# T wg     : ', Tg)
        print('# T w      : ', T)
        print('# N_eff wg : ', N_effg)
        print('# N_eff w  : ', N_eff)
        print('# min X2   : ', np.min(X2))
        print('# max pi   : ', np.max(pi))
        print('#-----------')
        
        self.calc_M_and_W(theta, w, wg)
        
        # transform
        thetas = np.ndarray(theta.shape)
        for iTheta in range(theta.shape[0]):
            thetas[iTheta,:] = np.dot(self.Winv,theta[iTheta,:]-self.M)
            
        # get sample weight
        buildW, buildWw = self.getBuildWAndWw(thetas, X2, T, pi, w, wg)
        
        # init kernel info
        self.kernels = []
        self.kernelWeight = []
        
        # exclude unimportant kernels
        argsortWw = np.argsort(buildWw)
        included_kernels = argsortWw[np.cumsum(np.sort(buildWw))>0.01]
        
        bins = np.ndarray((self.n_par, 2))
        
        for i_par in range(self.n_par):
            argsort_theta = np.argsort(thetas[:,i_par])
            sort_theta = thetas[argsort_theta, i_par]
            sort_wg = np.cumsum(wg[argsort_theta])
            sort_wg = sort_wg / sort_wg[-1]
            
            sort_w = np.cumsum(w[argsort_theta])
            sort_w = sort_w / sort_w[-1]
            
            bins[i_par, :] = sort_theta[[np.min([np.searchsorted(sort_w, 0.25),np.searchsorted(sort_wg, 0.25)]), 
                                         np.max([np.searchsorted(sort_w, 0.75),  np.searchsorted(sort_wg, 0.75)])]]
            
            
        #bins = bins + np.array([-0.5, 0.5]) * np.diff(bins)   

        #bins = np.array([[np.min(thetas[:,i]), np.max(thetas[:,i])] for i in range(self.n_par)])
    
        # create kernels
        for i_kernel in included_kernels:
            x_p_dist = self.fit_kernel(thetas, buildW[i_kernel], bins)
            if np.sum(x_p_dist[1])>0:
                self.kernels.append(x_p_dist)
                self.kernelWeight.append(buildWw[i_kernel])
            
        self.kernelWeight = self.kernelWeight / np.sum(self.kernelWeight )
        self.kernelWeightCumSum = np.cumsum(self.kernelWeight)
        self.kernelWeightCumSum = self.kernelWeightCumSum  / self.kernelWeightCumSum [-1]
        self.n_kernel = len(self.kernels)
        
    def fit_kernel(self, theta, w, bins):
        
        # preallocate
        hi = np.zeros((self.n_par, self.n_bins+1))
        x_dist = np.zeros((self.n_par, self.n_bins+1))
        
        if np.sum(w)==0:
            return (np.array([-np.Inf, np.Inf]), np.zeros(2))
        
        # create histogram
        for i_par in range(self.n_par):
            
            argsort_theta = np.argsort(theta[:,i_par])
            sort_w = np.cumsum(w[argsort_theta])
            sort_w = sort_w / sort_w[-1]
            
            r = [np.min([theta[argsort_theta[np.searchsorted(sort_w, 0.05)],   i_par], bins[i_par,0]]), 
                 np.max([theta[argsort_theta[np.searchsorted(sort_w, 0.95)], i_par], bins[i_par,-1]])]
            
            fac = np.min([(bins[i_par,1]-bins[i_par,0]), 1])/(r[1]-r[0])
            
            r = r + np.array([-0.5, 0.5]) * (r[1]-r[0])
            
            hi[i_par, 1:], x_dist[i_par, :] = np.histogram(theta[:,i_par], weights=w, bins=self.n_bins, range=r)
            
            #delta = x_dist[i_par, 27] - x_dist[i_par, 26]
            #rs = [r[0] - 25*delta, r[1] +25*delta]
            #x_dist[i_par, :] = np.linspace(rs[0], rs[1], self.n_bins+1)
            
            conv = np.log(np.linspace(-0.99,0.99,201)*fac+1)*get_effective_number_of_samples(w)**0.2
            conv = np.exp(-conv**2)[::-1]

            hi[i_par,:]=np.convolve(hi[i_par,:], conv, 'same')

        p_dist = np.cumsum(hi,axis=1)
        p_dist = p_dist / p_dist[:,[-1]]
            
        x_dist[:, 0] = -np.Inf
        x_dist[:, -1] = np.Inf
        
        p_dist[:, :2] = 0
        p_dist[:, -2:] = 1
        
        if np.any(np.isnan(x_dist)):
            stop
        if np.any(np.isnan(p_dist)):
            stop
        
        return (x_dist, p_dist)
        
    def calc_M_and_W(self, theta, w1, w2):
        self.M = np.average(theta,axis=0, weights=w1) 
        if theta.shape[0]>theta.shape[1]:
            eigvec = np.linalg.eig( np.cov( (theta-self.M).transpose(), aweights = w2 )  )
            self.W = eigvec[1]
        else:
            self.W = np.eye(theta.shape[1])
        self.Winv = np.linalg.inv(self.W)
        
        
    
    def score_gamma_weight(self,theta, X2, pi, Tmin=1, Tmax=1e6, logcorrection=None):
        # Determine temperature
        T = Tmin
        
        if logcorrection is None:
            logcorrection = np.zeros(X2.shape)

        wg = np.exp( (-X2-pi+logcorrection) / T -logcorrection)

        # newton raphson
        
        n_eff_target = self.n_par * self.sim_fac
        Ttrymin = 1
        Ttrymax = 100
        for i in range(5):
            Ttry = np.linspace(Ttrymin, Ttrymax, 10)
            n_eff_try = np.array([get_effective_number_of_samples(np.exp( (-X2-pi) / Ttry[i] )) for i in range(len(Ttry))])
            
            if n_eff_try[0]>n_eff_target:
                return 1, wg
            elif n_eff_try[-1]<n_eff_target:
                Ttrymax = Ttrymax * 10
            else:
                Ttrymin = Ttry[np.argmax(n_eff_try>n_eff_target)-1]
                Ttrymax = Ttry[np.argmax(n_eff_try>n_eff_target)]
            
        if n_eff_try[-1]<n_eff_target:
            T = Ttrymax
        else:
            T = np.interp(n_eff_target, n_eff_try, Ttry)
        wg = np.exp( (-X2-pi) / T)
        n_eff = get_effective_number_of_samples(wg)

        return T, wg  
    
    def getBuildWAndWw(self, theta, X2, pi, T, w1, ws):
        
        w = np.array(w1)
        
        wmax = [0.1, 0.5, 1]
        nw = [100, 200, 300]
        
        for i1, i2 in zip(wmax, nw):
            if len(w)>i2:
                maxW = np.max(w[:-i2])
                ws = w[-i2:]
                ws[ws>i1*maxW]=i1*maxW
                w[-i2:]=ws
        
        
        nKernels = np.max([5, int(np.min([get_effective_number_of_samples(w)/2, 100])) ])
        
        w1 = [np.array(w)]
        
        self.w_k1  = [1]
        #return (w1, self.w_k1)

        
        # Train KMEans
        from sklearn.cluster import KMeans  #For applying KMeans
        kmeans = KMeans(n_clusters=nKernels, n_init=10, random_state=0, max_iter=1000)

        idx = w>0.01
        if np.sum(idx)<nKernels*2:
            idx = np.argsort(w)[::-1][:nKernels*2]

        kmeans.fit(theta[idx,:], sample_weight=w[idx])
        
       
        idx = np.zeros(theta.shape[0], dtype=int) - 1
        for i in range(theta.shape[0]):
            if w[i]/np.max(w)>1e-4 or len(w)<1000:
                idx[i] = kmeans.predict(theta[i,:].reshape(1,-1))
                
        del kmeans
        
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
        
        idxLow = self.w_k1>0.25
        while np.any(idxLow):
            self.w_k1[idxLow] = 0.25
            self.w_k1[idxLow==False] = self.w_k1[idxLow==False] / np.sum(self.w_k1[idxLow==False]) * (1-np.sum(self.w_k1[idxLow]))
            self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
            idxLow = self.w_k1>0.25+1e-3
        
        self.n_kernel = len(w1)
        self.K = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_k = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        
        buildWw = self.w_k1 
        print('number of kernels: ',len(self.w_k1 ))
        
        return (buildW, buildWw)   
    
    def get_samples(self, n_samples):
        """Get multiple samples."""
        return np.array([self.get_sample() for _ in range(n_samples)])
        
        
    def get_sample(self):
        """Get multiple samples."""
        u = np.random.random()
        i_kernel = np.searchsorted(self.kernelWeightCumSum, u)
        
        X_uni = np.random.random((self.n_par))
        sample = np.array([np.interp(X_uni[i], self.kernels[i_kernel][1][i,:], self.kernels[i_kernel][0][i,:]) for i in range(self.n_par)])
        return np.dot(self.W, sample) + self.M
    
    def score_samples(self, theta):
        log_score = np.zeros((theta.shape[0], self.n_kernel)) - np.Inf
        
        thetas = np.array(theta)
        for iTheta in range(thetas.shape[0]):
            thetas[iTheta,:] = np.dot(self.Winv,theta[iTheta,:]-self.M)
        
        for i_kernel in range(self.n_kernel):
            try:
                sample_score_loc = np.array([np.searchsorted(self.kernels[i_kernel][0][i,:], thetas[:,i]) for i in range(self.n_par)])
            except:
                sample_score_loc = np.array([np.searchsorted(self.kernels[i_kernel][0][i,:], thetas[:,i]) for i in range(self.n_par)])
            q_ipar_top = np.array([ ( self.kernels[i_kernel][1][i,sample_score_loc[i,:]] - self.kernels[i_kernel][1][i,sample_score_loc[i,:]-1] )  for i in range(self.n_par)])
            q_ipar_bot = np.array([ (self.kernels[i_kernel][0][i,sample_score_loc[i,:]] - self.kernels[i_kernel][0][i,sample_score_loc[i,:]-1])  for i in range(self.n_par)])
            
            q_ipar_top[np.isinf(q_ipar_bot)] = 0
            q_ipar_bot[np.isinf(q_ipar_bot)] = 1
            
            q_ipar = q_ipar_top / q_ipar_bot
            
            if np.any(np.isnan(q_ipar)):
                stop
            
            log_q_ipar = np.ndarray(q_ipar.shape)
            log_q_ipar[q_ipar==0] = -np.Inf
            log_q_ipar[q_ipar>0] = np.log(q_ipar[q_ipar>0])
            
            if np.any(np.isnan(log_q_ipar)):
                stop
            
            log_score[:, i_kernel] = np.sum(log_q_ipar, axis=0)
            
            if np.any(np.isnan(log_score)):
                stop
        
        score = np.zeros((theta.shape[0], self.n_kernel))
        score[np.isinf(log_score)==False] = np.exp(log_score[np.isinf(log_score)==False])
        score = np.sum(self.kernelWeightCumSum*score, axis=1)
        
        log_score = np.zeros((theta.shape[0])) - np.Inf
        
        if np.any(np.isnan(log_score)):
                stop
        
        log_score[score>0] = np.log(score[score>0])
        
        if np.any(np.isnan(log_score)):
            stop
        
        return log_score
    
#%%
class kernel1a(kernel1):
    def fit_kernel(self, theta, w, bins):
        
        # preallocate
        hi = np.zeros((self.n_par, self.n_bins+1))
        x_dist = np.zeros((self.n_par, self.n_bins+1))
        
        if np.sum(w)==0:
            return (np.array([-np.Inf, np.Inf]), np.zeros(2))
        
        # create histogram
        for i_par in range(self.n_par):
            
            argsort_theta = np.argsort(theta[:,i_par])
            sort_w = np.cumsum(w[argsort_theta])
            sort_w = sort_w / sort_w[-1]
            
            r = [np.min([theta[argsort_theta[np.searchsorted(sort_w, 0.05)],   i_par], bins[i_par,0]]), 
                 np.max([theta[argsort_theta[np.searchsorted(sort_w, 0.95)], i_par], bins[i_par,-1]])]
            
            fac = np.min([(bins[i_par,1]-bins[i_par,0]), 1])/(r[1]-r[0])
            
            
            r = r + np.array([-0.5, 0.5]) * (r[1]-r[0])
            
            hi[i_par, 1:], x_dist[i_par, :] = np.histogram(theta[:,i_par], weights=w, bins=self.n_bins, range=r)
            
            #delta = x_dist[i_par, 27] - x_dist[i_par, 26]
            #rs = [r[0] - 25*delta, r[1] +25*delta]
            #x_dist[i_par, :] = np.linspace(rs[0], rs[1], self.n_bins+1)
            
            n=1+2*int( (np.max([0, (len(w)-300)])*25 + 500000) / ( (np.max([0, (len(w)-300)]) + 5000)))
          
            conv = np.log(np.linspace(-0.99,0.99,n)*fac+1)*1*get_effective_number_of_samples(w)**0.2
            conv = np.exp(-conv**2)[::-1]
            
            hi[i_par,:]=np.convolve(hi[i_par,:], conv, 'same')
           
        p_dist = np.cumsum(hi,axis=1)
        p_dist = p_dist / p_dist[:,[-1]]
            
        x_dist[:, 0] = -np.Inf
        x_dist[:, -1] = np.Inf
        
        p_dist[:, :2] = 0
        p_dist[:, -2:] = 1
        
        if np.any(np.isnan(x_dist)):
            stop
        if np.any(np.isnan(p_dist)):
            stop
        
        return (x_dist, p_dist)
    
    def getBuildWAndWw(self, theta, X2, pi, T, w1, ws):
        
        w = np.array(w1)
        
        wmax = [0.1, 0.5, 1]
        nw = [100, 200, 300]
        
        for i1, i2 in zip(wmax, nw):
            if len(w)>i2:
                maxW = np.max(w[:-i2])
                ws = w[-i2:]
                ws[ws>i1*maxW]=i1*maxW
                w[-i2:]=ws
        
        if len(w)<600:
            w[:300] = np.max(w[:300])
        
        nKernels = np.max([10, int(np.min([get_effective_number_of_samples(w)/20+1, 100])) ])
        
        w1 = [np.array(w)]
        
        self.w_k1  = [1]
        if nKernels == 1:
            return (w1, self.w_k1)

        
        # Train KMEans
        from sklearn.cluster import KMeans  #For applying KMeans
        kmeans = KMeans(n_clusters=nKernels, n_init=10, random_state=0, max_iter=1000)

        idx = w>0.01
        if np.sum(idx)<nKernels*2:
            idx = np.argsort(w)[::-1][:nKernels*2]

        kmeans.fit(theta[idx,:], sample_weight=w[idx])
        
       
        idx = np.zeros(theta.shape[0], dtype=int) - 1
        for i in range(theta.shape[0]):
            if w[i]/np.max(w)>1e-4 or len(w)<1000:
                idx[i] = kmeans.predict(theta[i,:].reshape(1,-1))
                
        del kmeans
        
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
        
        idxLow = self.w_k1>1/nKernels*2
        while np.any(idxLow):
            self.w_k1[idxLow] = 1/nKernels*2
            self.w_k1[idxLow==False] = self.w_k1[idxLow==False] / np.sum(self.w_k1[idxLow==False]) * (1-np.sum(self.w_k1[idxLow]))
            self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
            idxLow = self.w_k1>nKernels/4+1e-3
        
        self.n_kernel = len(w1)
        self.K = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_k = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        
        buildWw = self.w_k1 
        print('number of kernels: ',len(self.w_k1 ))
        
        return (buildW, buildWw)  
    
class kernel1b(kernel1a):
    def getBuildWAndWw(self, theta, X2, pi, T, w1, ws):
        
        w = np.array(w1)
        
        wmax = [0.25, 0.5, 1]
        nw = [100, 200, 300]
        
        for i1, i2 in zip(wmax, nw):
            if len(w)>i2:
                maxW = np.max(w[:-i2])
                ws = w[-i2:]
                ws[ws>i1*maxW]=i1*maxW
                w[-i2:]=ws

        if np.sum(w>0)>20:   
            # give the best 200 samples the same weigth
            asortw = np.argsort(w)
            w[asortw[-20:]] = w[asortw[-20]]
        elif len(w)>20:
            w[w>0]=0
            asort = np.argsort(X2)
            w[asort[:20]] = 1
        
        w = w / np.sum(w)
        
        nKernels = np.max([10, int(np.min([get_effective_number_of_samples(w)/20+1, 100])) ])
        
        w1 = [np.array(w)]
        
        self.w_k1  = [1]
        if nKernels == 1:
            return (w1, self.w_k1)

        
        # Train KMEans
        from sklearn.cluster import KMeans  #For applying KMeans
        kmeans = KMeans(n_clusters=nKernels, n_init=10, random_state=0, max_iter=1000)

        idx = w>0.01
        if np.sum(idx)<nKernels*2:
            idx = np.argsort(w)[::-1][:nKernels*2]

        kmeans.fit(theta[idx,:], sample_weight=w[idx])
        
       
        idx = np.zeros(theta.shape[0], dtype=int) - 1
        for i in range(theta.shape[0]):
            if w[i]/np.max(w)>1e-4 or len(w)<1000:
                idx[i] = kmeans.predict(theta[i,:].reshape(1,-1))
                
        del kmeans
        
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
        
        idxLow = self.w_k1>1/nKernels*2
        while np.any(idxLow):
            self.w_k1[idxLow] = 1/nKernels*2
            self.w_k1[idxLow==False] = self.w_k1[idxLow==False] / np.sum(self.w_k1[idxLow==False]) * (1-np.sum(self.w_k1[idxLow]))
            self.w_k1 = np.array(self.w_k1)/np.sum(self.w_k1)
            idxLow = self.w_k1>nKernels/4+1e-3
        
        self.n_kernel = len(w1)
        self.K = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        self.w_k = [[[] for _ in range(self.n_par)] for _ in range(self.n_kernel)]
        
        buildWw = self.w_k1 
        print('number of kernels: ', 
              len( self.w_k1 ) )
        
        return (buildW, buildWw)
    
class kernel1c(kernel1b):

    #def calc_M_and_W(self, theta, w, wg):
        #asort_wg = np.argsort(wg)
        #w1 = np.array(wg)
        #w1[asort_wg[-20:]]=w1[asort_wg[-20]]
        #super().calc_M_and_W(theta, w, w1)
        
    def score_gamma_weight(self, theta, X2, pi, Tmin=1, Tmax=1e6, logcorrection=None):
        T, wg  = super().score_gamma_weight(theta, X2, pi, Tmin=Tmin, Tmax=Tmax, logcorrection=None)
        
        asort_wg = np.argsort(wg)
        w1 = np.array(wg)
        w1[asort_wg[-20:]]=w1[asort_wg[-20]]
        
        return T, w1 
        
#%%
def get_effective_number_of_samples(w):
    if np.sum(w**2)<1e-300:
        return 1
    return np.sum(w)**2 / np.sum(w**2)