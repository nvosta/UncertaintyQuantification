import numpy as np
import time
from sklearn.neighbors import KernelDensity

# -*- coding: utf-8 -*-
class Sampler:
    """Basic functions for sampler."""

    dummy_samples = []
    samplesModel = []
    sampler_generation_time = 0

    def get_samples(self, nSamples, max_time=np.Infinity):
        """Get a matrix with nSamples."""
        samples = []
        tic = time.time()
        while len(samples) < nSamples:
            samples.append(self.get_sample())
            if time.time()-tic > max_time:
                return np.array(samples)
        return np.array(samples)

    def score_samples(self, theta):
        """Score Samples."""
        return np.array([self.score_sample(theta[i, :]) for i in range(
            theta.shape[0])])[:,0]

    def load_dummy_samples(self, nSamples, max_time=np.Infinity):
        """Load dummy samples for postprocessing."""
        self.dummy_samples = (
            self.get_samples(nSamples, max_time=max_time))

    def run_dummy_samples(self, model, parameters):
        """Load dummy samples for postprocessing."""
        self.samplesModel = []
        for theta in self.dummy_samples:
            model_instance = model.getInstance()
            parameters.setX(model_instance, theta)
            model_instance.run()
            if model_instance.getIsStable():
                self.samplesModel.append(model_instance.getPdict())

class SamplerSemiUniform(Sampler):
    def __init__(self, prior_data, **kwargs):

        self.prior_theta = prior_data['theta_info']['theta']
        self.prior_X2 = prior_data['theta_info']['X2']
        self.prior_pi = np.log(np.sum(np.exp(prior_data['theta_info']['pi']), axis=1))

        # calculate temperature
        nSims = 20
        s = 0
        temperature = 1
        maxTemp = 1e6
        while s<nSims and temperature < maxTemp:
            sample_weigth = np.exp(-self.prior_X2/temperature - self.prior_pi)
            s = np.sum(sample_weigth)/np.max(sample_weigth)
            if s<nSims:
                temperature = np.min([maxTemp, temperature*10])
        self.temperature = temperature
        # calculate distance allowed from points
        self.std = 0.5*(np.max(self.prior_theta,axis=0)-np.min(self.prior_theta,axis=0))
        self.sample_weigth = sample_weigth / np.max(sample_weigth)

        #remove unimportant samples
        idx = self.sample_weigth>0.01
        self.prior_theta = self.prior_theta[idx,:]
        self.prior_X2 = self.prior_X2[idx]
        self.prior_pi = self.prior_pi[idx]

        # set abolute lower and upper bound
        self.lower_bound = np.min(self.prior_theta-self.std.reshape(1,-1)*self.sample_weigth.reshape(-1,1),axis=0)
        self.upper_bound = np.max(self.prior_theta+self.std.reshape(1,-1)*self.sample_weigth.reshape(-1,1),axis=0)



        # calculate score
        self.score = 1
        tTest=100000
        nTest=0
        i = 0
        while i < tTest or nTest<1:
            i=i+1
            sample = np.random.random(self.prior_theta.shape[1])
            sample = self.lower_bound + (
                self.upper_bound-self.lower_bound)*sample
            if self.score_sample(sample)>0:
                nTest=nTest+1
        self.score = i/nTest


    def get_sample(self, retN=False):
        nTest=0
        while True:
            nTest=nTest+1
            sample = np.random.random(self.prior_theta.shape[1])
            sample = self.lower_bound + (
                self.upper_bound-self.lower_bound)*sample

            if np.isinf(self.score_sample(sample))==False:
                if retN:
                    return sample, nTest
                return sample

    def score_sample(self, theta):
        d = np.abs((theta-self.prior_theta)/self.std)

        if np.any(np.sum(d<self.sample_weigth.reshape(-1,1),axis=1)==d.shape[1]):
            return self.score
        else:
            return -np.Infinity

    def get_theta_sample_weights(self, theta, X2, prior_data=[], returnTemperature=False, maxTemperature=None, use_temperature=None):
        sample_weight = np.ones((X2.shape[0]))
        if returnTemperature:
            return theta, sample_weight, np.Infinity
        return theta, sample_weight



class SamplerUniform(Sampler):
    """Uniform sampling."""

    options = {'lower_bound': [], 'upper_bound': []}
    kdeSettings = {}

    def __init__(self, parameters, kdeSettings={}, **kwargs):
        self.options.update(kwargs)
        self.parameters = parameters
        self.kdeSettings.update(kdeSettings)
        self.temperature = self.kdeSettings['max_temperature']

        if len(self.options['lower_bound']) == 0:
            self.options['lower_bound'] = np.zeros(len(self.parameters))
        if len(self.options['upper_bound']) == 0:
            self.options['upper_bound'] = np.ones(len(self.parameters))

    def get_sample(self):
        """Get a single sample."""
        sample = np.random.random(len(self.parameters))
        sample = self.options['lower_bound'] + (
            self.options['upper_bound']-self.options['lower_bound'])*sample
        return sample

    def score_sample(self, theta):
        """Score Sample."""
        if np.any(theta < self.options['lower_bound']) or np.any(
                theta > self.options['upper_bound']):
            return -np.Infinity
        return np.log(np.prod(1/(
            self.options['upper_bound'] - self.options['lower_bound'])))

    def score_samples(self, theta):
        score = np.ones((theta.shape[0]))
        for iPar in range(len(self.parameters)):
            r = self.options['upper_bound'][iPar] - self.options['lower_bound'][iPar]
            score[:]=score*1/np.prod(r)
        score = np.log(score)

        # correct for boundaries
        score[np.any(theta<self.options['lower_bound'], axis=1)]=-np.Infinity
        score[np.any(theta>self.options['upper_bound'], axis=1)]=-np.Infinity

        return score

    def get_theta_sample_weights(self, theta, X2, prior_data=[], returnTemperature=False, maxTemperature=None, use_temperature=None):
        sample_weight = np.ones((X2.shape[0],  len(self.kdeSettings['combineKDE'])))
        if returnTemperature:
            return theta, sample_weight, np.Infinity
        return theta, sample_weight


class SamplerKDE(Sampler):
    """Sampling from KDE."""


    def initKDEsettings(self):
        self.kdeSettings = {'kernel': 'gaussian',
                   "minNSims": 1,
                   'bandwidth_selection': 'Scott',
                   'bandwidht_fac': 1,
                   'anneal_X2': False,
                   'anneal_logcombi': False,
                   'anneal_logcombi1': False,
                   'Yfac': 1,
                   'bandwidht_fac': 1,
                   'combine_pi': False,
                    'min_temperature_fac': 0.9,
                    'min_temperature': 1,
                    'max_temperature': 1e3,
                    'add_to_proposal': 0,
                    'average_annealed_with_normal_distribution': False,
                    'annealed_weight':2,
                    'average_annealed_with_normal_distribution_multiple':[],
                    'average_annealed_with_normal_distribution_multiple_reduce_factor': False,
                    'annealed_weight_multiple': [],
                    'annealed_weight_isTempFac': [],
                    'annealed_use_X2':[],
                    'fac_use_in_kde': 0}
        self.kde = []
        self.kde_parameters = []

        self.nSims = 0
        self.nSimsEff = 0

    def __init__(self, parameters, theta, X2, kdeSettings=None, reuse_sample_weight=False):
        self.initKDEsettings()









        self.parameters = parameters
        self.kdeSettings.update(kdeSettings)
        self.temperature = np.ones(len(self.kdeSettings['combineKDE']))

        self.fit(theta, X2, reuse_sample_weight=reuse_sample_weight)

    def fit(self, theta, X2, prior_data, reuse_sample_weight=False ):
        """Fit kernels."""
        iParDone = []
        self.kde = []
        self.kde_parameters = []

        # get sample_weight
        if len(prior_data['iter_info']['temperature'])>1:
            maxTemperature = self.kdeSettings['max_temperature']#np.min(np.array(prior_data['iter_info']['temperature'][:-1]))
            minTemperature = self.kdeSettings['min_temperature_fac']*np.array(prior_data['iter_info']['temperature'][-1])
            minTemperature = np.max([minTemperature, self.kdeSettings['min_temperature']])
        else:
            minTemperature = self.kdeSettings['max_temperature']
            maxTemperature = self.kdeSettings['max_temperature']
        theta, sample_weight_all = self.get_theta_sample_weights(
            theta, X2, prior_data=prior_data,
            maxTemperature=maxTemperature, minTemperature=minTemperature)

        temperature = self.temperature

        if reuse_sample_weight:#np.sum(prior_data['theta_info']['sample_weight'][:,-1])>0:
            sample_weight_all = prior_data['theta_info']['sample_weight'][:,-1]
        elif self.kdeSettings['average_annealed_with_normal_distribution']:
            _, sample_weight_all1 = self.get_theta_sample_weights(
            theta, X2, prior_data=prior_data,
            use_temperature=1)

            sample_weight_all = sample_weight_all / np.max(sample_weight_all)
            fac = np.max([1, self.kdeSettings['minNSims']/np.sum(sample_weight_all) ])
            # in beginnnig, sample_weight_all1 is nan, in that case, use
            if np.any(np.isnan(sample_weight_all1)):
                print('Real distribution is nan, improve distribution based on history without information of goodness')
            else:
                sample_weight_all = sample_weight_all/np.sum(sample_weight_all)*fac*self.kdeSettings['annealed_weight'] + sample_weight_all1/np.sum(sample_weight_all1)

        elif len(self.kdeSettings['average_annealed_with_normal_distribution_multiple'])>0:
            sample_weight_all = self.get_average_annealed_sample_weight_with_normal_distribution_multiple(
                theta, X2, prior_data, sample_weight_all)



        if 'combineKDE' in self.kdeSettings:
            for iKDE in range(len(self.kdeSettings['combineKDE'])):
                # select parameters for this kde
                includeIpar = np.array(self.kdeSettings['combineKDE'][iKDE])
                iParDone = iParDone + list(
                    self.kdeSettings['combineKDE'][iKDE])

                theta_included = theta[:, includeIpar]
                sample_weight = np.array(sample_weight_all)

                # remove unimportant points
                argsort = np.argsort(sample_weight)
                cumsort = np.cumsum(sample_weight[argsort])
                idx = argsort[cumsort>0.01*cumsort[-1]]
                if len(idx)<5:
                    idx = argsort[-5:]
                if len(sample_weight)<2000:#np.sum(idx)>self.kdeSettings['minNSims']+1:
                    idx = sample_weight>1e-100
                sample_weight = sample_weight[idx]
                theta_included = theta_included[idx,:]
                theta_included = theta_included / self.kde_std[includeIpar]

                # fit distribution
                kde = self.fit_single_distribution(
                    theta=theta_included,
                    sample_weight=sample_weight, bandwidth=self.sampler_bandwidth)
                # distribution to memory
                self.kde.append(kde)
                self.kde_parameters.append(includeIpar)
                #print('Fit KDE finished, summed samples:',
                #      (np.sum(sample_weight)/np.max(sample_weight)))
                self.nSims = len(theta)
                self.nSimsEff = np.sum(sample_weight)/np.max(sample_weight)
        # ToDo: Single kdes


        self.temperature = temperature

    def get_average_annealed_sample_weight_with_normal_distribution_multiple(self, theta, X2, prior_data, sample_weight_all=[]):
        sample_weight_total = np.zeros(len(theta))

        X2w = X2 / self.kdeSettings['Yfac']
        w = np.array(prior_data['iter_info']['nSims'])/ np.sum(prior_data['iter_info']['nSims'])
        w = w.reshape((1,-1))
        weight_prior = np.sum(w * np.exp(
            np.array(prior_data['theta_info']['pi'])), axis=1)
        weight_prior[weight_prior<1e-300] = 1e-300
        weight_prior = np.log(weight_prior)

        if ('annealed_split_distribution' in self.kdeSettings and
            self.kdeSettings['annealed_split_distribution']):
                sample_weigth_list = []

        for iTemp in range(len(self.kdeSettings['average_annealed_with_normal_distribution_multiple'])):
            useTemp = self.kdeSettings['average_annealed_with_normal_distribution_multiple'][iTemp]

            if len(self.kdeSettings['annealed_weight_isTempFac'])>iTemp and self.kdeSettings['annealed_weight_isTempFac'][iTemp]:
                if len(prior_data['iter_info']['temperature'][:])>1:
                    useTemp = useTemp * np.min([prior_data['iter_info']['temperature'][:-1]])
                else:
                    useTemp = useTemp * np.max(self.kdeSettings['average_annealed_with_normal_distribution_multiple'])


            if self.kdeSettings['average_annealed_with_normal_distribution_multiple_reduce_factor']:
                if len(prior_data['iter_info']['temperature'][:])>1:
                    useTemp = np.min([useTemp, np.min([prior_data['iter_info']['temperature'][:-1]])])


            useFac = self.kdeSettings['annealed_weight_multiple'][iTemp]

            if len(self.kdeSettings['annealed_use_X2'])>iTemp and self.kdeSettings['annealed_use_X2'][iTemp]>-1:
                useX2 = prior_data['theta_info']['mX2'][:,self.kdeSettings['annealed_use_X2'][iTemp]]
            else:
                useX2 = np.array(X2w)

            if useTemp>0:
                if 'annealed_allow_below_one' in self.kdeSettings and len(self.kdeSettings['annealed_allow_below_one'])>iTemp and not self.kdeSettings['annealed_allow_below_one'][iTemp]:
                    useTemp = np.max([useTemp, 1])
                sample_weight_all1 = np.exp( (- useX2/useTemp  - weight_prior) )
                if np.sum(sample_weight_all1)>1e-100:
                    sample_weight_all1 = sample_weight_all1/np.sum(sample_weight_all1)
            elif self.kdeSettings['average_annealed_with_normal_distribution_multiple'][iTemp]==-2:
                sample_weight_all1 = sample_weight_all1*0+1/len(sample_weight_all1)
            else:
                sample_weight_all1 = sample_weight_all
                sample_weight_all1 = sample_weight_all1/np.sum(sample_weight_all1) * np.max([1, self.kdeSettings['minNSims']/np.sum(sample_weight_all1) ])



            if ('annealed_split_distribution' in self.kdeSettings and
                self.kdeSettings['annealed_split_distribution']):
                sample_weigth_list.append(sample_weight_all1)
            elif np.sum(sample_weight_all1)>1e-300:
                sample_weight_total = sample_weight_total + sample_weight_all1*useFac

        if ('annealed_split_distribution' in self.kdeSettings and
                self.kdeSettings['annealed_split_distribution']):
            return sample_weigth_list
        #else
        sample_weight_all = sample_weight_total / np.max(sample_weight_total)
        return sample_weight_all





    def fit_single_distribution(self, theta, sample_weight, bandwidth=1):
        kde = KernelDensity(bandwidth=bandwidth,
                                    kernel=self.kdeSettings['kernel'])
        kde.fit(theta, sample_weight=sample_weight)
        return kde

    def get_bandwidth(self, nSim, nPar, std=1):
        """Get bandwidht debending on kdeSettings->bandwidth_selection."""
        if self.kdeSettings['bandwidth_selection'] in ['Silverman', 'silverman_temp1']:
            return std * (4/(nSim*(nPar+2)))**(2/(nPar+4))
        if self.kdeSettings['bandwidth_selection'] in ['std/4', 'window/4']:
            return std/4
        elif self.kdeSettings['bandwidth_selection'] in ['2wstd']:
            return 2
        elif self.kdeSettings['bandwidth_selection'] in ['wstd']:
            return 1
        elif self.kdeSettings['bandwidth_selection'] in ['wIQR']:
            return 0.9*nSim**(-1/(4+nPar))
        else:
            return nSim**(-1/(4+nPar))

    def get_theta_sample_weights(self, theta, X2, use_temperature=None, maxTemperature=None, minTemperature=None):
        """
        Get theta and sample_weigths.

        Removes infinity, nan, and weigths sample_weights to minNSims
        """
        # fit prior
        print('Fit KDE Prior')
        X2 = X2 / self.kdeSettings['Yfac']
        # bandwidth = len(X2)**(-1/4+theta.shape[1])
        stdTheta = np.std(theta, axis=0)
        bandwidth = self.kdeSettings['bandwidht_fac'] * self.get_bandwidth(len(X2), theta.shape[1])
        kde_prior = KernelDensity(bandwidth=bandwidth,
                                  kernel=self.kdeSettings['kernel'])
        kde_prior.fit(theta/stdTheta)
        print('Fit KDE Prior finished')

        # remove samples with nan and inf
        idx = np.isnan(X2) == False
        X2 = X2[idx]
        theta = theta[idx, :]
        idx = np.isinf(X2) == False
        X2 = X2[idx]
        theta = theta[idx, :]

        # Remove samples
        #idx = X2 < np.min(X2) + 10
        #minSims = np.max([100, self.kdeSettings['minNSims']*50])
        #if sum(idx) < minSims:
        #    idx = np.argsort(X2)[:minSims]
        #X2 = X2[idx]
        #theta = theta[idx, :]

        # get sample weigth
        weight_prior = kde_prior.score_samples(theta/stdTheta)
        print('KDE Prior score_samples finished')
        log_sample_weight = -X2 - weight_prior
        log_sample_weight = log_sample_weight - np.max(log_sample_weight)

        sample_weight = self.get_sample_weight_annealed(
            log_sample_weight, maxTemperature=maxTemperature, minTemperature=minTemperature)

        # remove samples with too low weight
        idx = np.argsort(sample_weight)
        cs = np.cumsum(sample_weight[idx])
        #idx = sample_weight > 0.0001 * np.max(sample_weight)
        idx = idx[cs>0.05]
        sample_weight = sample_weight[idx]
        theta = theta[idx, :]

        return theta, sample_weight

    def calc_temperature(self, X2, weight_prior, maxTemperature=None,
                                   temperature=None, use_temperature=None, minTemperature=1):
        weight_factor = minTemperature
        sample_weight = np.zeros((X2.shape[0],  len(self.kdeSettings['combineKDE'])))

        iTry=0
        while iTry == 0 or (
                (np.any(np.sum(sample_weight,axis=0) < self.kdeSettings['minNSims']) and
                   np.any(weight_factor < maxTemperature)
                   and np.any(weight_factor > minTemperature)
                ) or (
                   np.any(weight_factor > 1)
                   and np.any(np.sum(sample_weight,axis=0) > 1.05*self.kdeSettings['minNSims'])
                )):
            iTry = iTry + 1
            if self.kdeSettings['anneal_X2']:
                X2w = X2 / weight_factor
            else:
                X2w = X2
            if self.kdeSettings['anneal_logcombi1']:
                sample_weight = np.exp( (- X2w  - weight_prior + np.max(weight_prior)) / weight_factor)
            elif self.kdeSettings['anneal_logcombi']:
                sample_weight = np.exp( (- X2w  - weight_prior) / weight_factor)
            else:
                sample_weight = np.exp( (- X2w  - weight_prior) )

            if  np.max(sample_weight, axis=0)>0:
                sample_weight = sample_weight / np.max(sample_weight, axis=0)

            noChange = 0
            s = np.sum(sample_weight)
            if weight_factor<maxTemperature and s < self.kdeSettings['minNSims']:
                if s < 0.001*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 10
                elif s < 0.01*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 5
                elif s < 0.05*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 2.5
                elif s < 0.5*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 1.5
                elif s < self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 1.05
                else:
                    noChange = noChange + 1
                if weight_factor>maxTemperature:
                    weight_factor=maxTemperature
            elif weight_factor>1:
                if s > 1.1*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor / 1.01
                else:
                    noChange = noChange + 1

                if weight_factor<1:
                    weight_factor=1
                elif weight_factor<minTemperature:
                    weight_factor=minTemperature
                    noChange = noChange + 1

            else:
                noChange = noChange + 1
            if noChange>=1 or iTry>1e4:
                break

        #print('weight_factor2: ', weight_factor)
        print(np.sum(sample_weight, axis=0))

        self.temperature = weight_factor
        return self.temperature

    def get_final_sample_weight(self, theta, X2, prior_data,
                                maxTemperature=None, minTemperature=None, reuse_sample_weight=False):
        theta, sample_weight_all = self.get_theta_sample_weights(
            theta, X2, prior_data=prior_data,
            maxTemperature=maxTemperature, minTemperature=minTemperature)

        if reuse_sample_weight:#np.sum(prior_data['theta_info']['sample_weight'][:,-1])>0:
            sample_weight_all = prior_data['theta_info']['sample_weight'][:,-1]
        elif self.kdeSettings['average_annealed_with_normal_distribution']:
            _, sample_weight_all1 = self.get_theta_sample_weights(
            theta, X2, prior_data=prior_data,
            use_temperature=1)

            sample_weight_all = sample_weight_all / np.max(sample_weight_all)
            fac = np.max([1, self.kdeSettings['minNSims']/np.sum(sample_weight_all) ])
            # in beginnnig, sample_weight_all1 is nan, in that case, use
            if np.any(np.isnan(sample_weight_all1)):
                print('Real distribution is nan, improve distribution based on history without information of goodness')
            else:
                sample_weight_all = sample_weight_all/np.sum(sample_weight_all)*fac*self.kdeSettings['annealed_weight'] + sample_weight_all1/np.sum(sample_weight_all1)

        elif len(self.kdeSettings['average_annealed_with_normal_distribution_multiple'])>0:
            sample_weight_all = self.get_average_annealed_sample_weight_with_normal_distribution_multiple(
                theta, X2, prior_data, sample_weight_all)

        return theta, sample_weight_all


    def get_sample_weight_annealed(self, X2, weight_prior, maxTemperature=None,
                                   temperature=None, use_temperature=None, minTemperature=1):
        if maxTemperature is None:
            maxTemperature = 1e3

        if temperature is not None and len(temperature)>0:
            weight_factor = temperature
            if self.kdeSettings['anneal_X2']:
                X2w = X2 / weight_factor
            else:
                X2w = X2
            if self.kdeSettings['anneal_logcombi1']:
                sample_weight = np.exp( (- X2w  - weight_prior + np.max(weight_prior)) / weight_factor)
            elif self.kdeSettings['anneal_logcombi']:
                sample_weight = np.exp( (- X2w  - weight_prior) / weight_factor)
            else:
                sample_weight = np.exp( (- X2w  - weight_prior) )

            sample_weight = sample_weight / np.max(sample_weight)
            return sample_weight


        if use_temperature is not None:
            weight_factor = use_temperature
            if self.kdeSettings['anneal_X2']:
                X2w = X2 / weight_factor
            else:
                X2w = X2
            if self.kdeSettings['anneal_logcombi1']:
                sample_weight = np.exp( (- X2w  - weight_prior + np.max(weight_prior)) / weight_factor)
            elif self.kdeSettings['anneal_logcombi']:
                sample_weight = np.exp( (- X2w  - weight_prior) / weight_factor)
            else:
                sample_weight = np.exp( (- X2w  - weight_prior) )

            if np.max(sample_weight) == 0: # temperature too low to calculate
                sample_weight[np.argmax((- X2w  - weight_prior))] = 1
            else:
               sample_weight = sample_weight / np.max(sample_weight, axis=0)
            return sample_weight

        if not minTemperature:
            minTemperature = 1

        weight_factor = minTemperature
        sample_weight = np.zeros((X2.shape[0],  len(self.kdeSettings['combineKDE'])))

        iTry=0
        while iTry == 0 or (
                (np.any(np.sum(sample_weight,axis=0) < self.kdeSettings['minNSims']) and
                   np.any(weight_factor < maxTemperature)
                   and np.any(weight_factor > minTemperature)
                ) or (
                   np.any(weight_factor > 1)
                   and np.any(np.sum(sample_weight,axis=0) > 1.05*self.kdeSettings['minNSims'])
                )):
            iTry = iTry + 1
            if self.kdeSettings['anneal_X2']:
                X2w = X2 / weight_factor
            else:
                X2w = X2
            if self.kdeSettings['anneal_logcombi1']:
                sample_weight = np.exp( (- X2w  - weight_prior + np.max(weight_prior)) / weight_factor)
            elif self.kdeSettings['anneal_logcombi']:
                sample_weight = np.exp( (- X2w  - weight_prior) / weight_factor)
            else:
                sample_weight = np.exp( (- X2w  - weight_prior) )

            if  np.max(sample_weight, axis=0)>0:
                sample_weight = sample_weight / np.max(sample_weight, axis=0)

            noChange = 0
            s = np.sum(sample_weight)
            if weight_factor<maxTemperature and s < self.kdeSettings['minNSims']:
                if s < 0.001*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 10
                elif s < 0.01*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 5
                elif s < 0.05*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 2.5
                elif s < 0.5*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 1.5
                elif s < self.kdeSettings['minNSims']:
                    weight_factor = weight_factor * 1.05
                else:
                    noChange = noChange + 1
                if weight_factor>maxTemperature:
                    weight_factor=maxTemperature
            elif weight_factor>1:
                if s > 1.1*self.kdeSettings['minNSims']:
                    weight_factor = weight_factor / 1.01
                else:
                    noChange = noChange + 1

                if weight_factor<minTemperature:
                    weight_factor=minTemperature
                    noChange = noChange + 1

            else:
                noChange = noChange + 1
            if noChange>=1 or iTry>1e4:
                break



        #print('weight_factor2: ', weight_factor)
        print(np.sum(sample_weight, axis=0))

        self.temperature = weight_factor

        return sample_weight


    def get_sample(self):
        """Get a single sample."""
        sample = np.ones(len(self.parameters))*np.nan
        for iKDE in range(len(self.kde)):
            s = self.kde[iKDE].sample()
            sample[self.kde_parameters[iKDE]] = s
        if np.any(np.isnan(sample)):
            raise Exception(
                "Not all kernels available")
        sample = sample * self.kde_std
        #sample[sample < 0] = -sample[sample < 0]
        #sample[sample > 1] = 2-sample[sample > 1]
        #if np.any(sample < 0) or np.any(sample > 1):
        #    sample = self.get_sample()
            #raise Exception('Sample outside boundaries')
            #sample = self.get_sample()
        return sample

    def score_sample(self, theta):
        # transform theta to kde-theta
        theta = theta / self.kde_std

        # sum score of individual kde's
        s = 0
        for iKDE in range(len(self.kde)):
            s = s + self.kde[iKDE].score_samples([theta[self.kde_parameters[iKDE]]])[0]

        # bandwidth standard deviation correction
        s = s - np.log(np.prod(self.kde_std))
        return s

    def score_samples(self, theta):
        """Score Samples."""
        if len(theta)==0:
            return []
        theta = theta / self.kde_std

        # sum score of individual kde's
        s = np.zeros(len(theta))
        for iKDE in range(len(self.kde)):
            s = s + self.kde[iKDE].score_samples(theta[:,self.kde_parameters[iKDE]])

        # bandwidth standard deviation correction
        s = s - np.log(np.prod(self.kde_std))
        return s

class SamplerKDEpriorBasedOnSampler(SamplerKDE):
    def __init__(self, parameters, prior_data, kdeSettings=None,
                 temperature=None,sampler_bandwidth=None, sampler_std=None,
                 reuse_sample_weight=False):
        self.initKDEsettings()
        self.parameters = parameters
        self.kdeSettings.update(kdeSettings)

        if temperature is None:
            self.temperature = np.ones(len(self.kdeSettings['combineKDE']))
            self.temperatuerIsCalculated=False
        else:
            self.temperature = temperature
            self.temperatuerIsCalculated=True

        self.prior_data = prior_data

        self.kde_std = sampler_std
        self.sampler_bandwidth = sampler_bandwidth

        theta = np.array(self.prior_data['theta_info']['theta'])
        X2 = np.array(self.prior_data['theta_info']['X2'])
        self.fit(theta, X2, self.prior_data, reuse_sample_weight=reuse_sample_weight)




    def score_samples(self,theta):
        theta = theta / self.kde_std
        score = np.zeros((theta.shape[0]))

        for iKDE in range(len(self.kde)):
            score[:] = score[:] + self.kde[iKDE].score_samples(theta[:,self.kde_parameters[iKDE]])
            score[:] = score[:] - np.log(np.prod(self.kde_std[self.kdeSettings['combineKDE'][iKDE]]))

        return score


    def get_theta_sample_weights(self, theta, X2, prior_data=[], maxTemperature=None, use_temperature=None, minTemperature=None):
        """Do not use theta, X2, but prior_data."""
        X2 = X2 / self.kdeSettings['Yfac']

        if prior_data==[]:
            prior_data = self.prior_data


        if len(X2) != len(prior_data['theta_info']['pi']):
            raise Exception('length not equal')

        # print(prior_data['theta_info']['pi'])
        #weight_prior = np.mean(
        #    np.array(prior_data['iter_info']['nSims']) * np.exp(
        #    np.array(prior_data['theta_info']['pi'])), axis=1)
        w = np.array(prior_data['iter_info']['nSims'])/ np.sum(prior_data['iter_info']['nSims'])
        w = w.reshape((1,-1))
        weight_prior = np.sum(w * np.exp(
            np.array(prior_data['theta_info']['pi'])), axis=1)
        weight_prior[weight_prior<1e-300] = 1e-300
        weight_prior = np.log(weight_prior)



        if self.temperatuerIsCalculated:
            sample_weight = self.get_sample_weight_annealed(X2, weight_prior,
                                                            temperature=self.temperature, use_temperature=use_temperature, maxTemperature=maxTemperature, minTemperature=minTemperature)
        else:
            sample_weight = self.get_sample_weight_annealed(X2, weight_prior, use_temperature=use_temperature, maxTemperature=maxTemperature, minTemperature=minTemperature)

        return theta, sample_weight



#%%
class SamplerInitPSO(Sampler):
    def __init__(self, parameters, psodata, model):
        self.parameters = parameters
        self.psodata = psodata
        self.model = model

        self.fit()

    def fit(self):
        self.kde = []
        theta = []
        for iParticle in range(len(self.psodata['swarm']['P'])):
            model_instance = self.model.getInstance()
            model_instance.setPdict(self.psodata['swarm']['P'][iParticle])
            theta.append(self.parameters.getX(model_instance))

        theta = np.array(theta)

        for iPar in range(len(self.parameters)):
            X = theta[:,iPar]
            kde = KernelDensity(bandwidth = 10*1/theta.shape[0]*np.std(X))
            kde.fit(X.reshape(-1, 1) )
            self.kde.append(kde)

    def get_sample(self):
        theta = np.ndarray(len(self.parameters))
        for iPar in range(len(self.parameters)):
            theta[iPar] = self.kde[iPar].sample()[0]
        return theta

    def score_sample(self, X):
        score = 0
        for iPar in range(len(self.parameters)):
            score = score + self.kde[iPar].score_samples([[X[iPar]]])
        return score


#%%
class SamplerKDEpriorBasedOnSamplerCustom(SamplerKDEpriorBasedOnSampler):
    def __init__(self, parameters, prior_data, kdeSettings=None,
                 temperature=None,sampler_bandwidth=None, sampler_std=None,
                 reuse_sample_weight=False):
        self.initKDEsettings()
        self.parameters = parameters
        self.kdeSettings.update(kdeSettings)

        if temperature is None:
            self.temperature = np.ones(len(self.kdeSettings['combineKDE']))
            self.temperatuerIsCalculated=False
        else:
            self.temperature = temperature
            self.temperatuerIsCalculated=True

        self.prior_data = prior_data

        self.kde_std = sampler_std
        self.sampler_bandwidth = sampler_bandwidth

        # Rotation/transformation matrix
        self.W = np.eye(len(parameters))
        self.M = np.zeros((len(parameters), len(parameters)))

        theta = np.array(self.prior_data['theta_info']['theta'])
        X2 = np.array(self.prior_data['theta_info']['X2'])
        self.fit(theta, X2, self.prior_data, reuse_sample_weight=reuse_sample_weight)

        self.prior_data={}




    def fit(self, theta, X2, prior_data, reuse_sample_weight=False ):
        # get sample_weight
        if len(prior_data['iter_info']['temperature'])>0:
            maxTemperature = self.kdeSettings['max_temperature']#np.min(np.array(prior_data['iter_info']['temperature'][:-1]))
            minTemperature = self.kdeSettings['min_temperature_fac']*np.min(np.array(prior_data['iter_info']['temperature']))
            minTemperature = np.min([np.max([minTemperature, self.kdeSettings['min_temperature']]),
                                     self.kdeSettings['min_temperature_fac']*self.kdeSettings['max_temperature']])
        else:
            minTemperature = self.kdeSettings['min_temperature']
            maxTemperature = self.kdeSettings['max_temperature']

        # settings
        bandwidth = self.sampler_bandwidth

        theta, sample_weight_all = self.get_final_sample_weight(theta, X2, prior_data, maxTemperature, minTemperature, reuse_sample_weight)

        # Calculate W here
        W = np.eye(len(self.parameters))
        P = np.eye(len(self.parameters))
        for iPar in range(len(self.parameters)):
            P[iPar,iPar] = 1/np.std(theta[:,iPar])
        M = np.mean(theta,axis=0)

        # correct for bad points, include best 50%
        idx = np.argsort(X2)[:int(theta.shape[0]/2)]
        if len(idx)>len(self.parameters)*10 and len(self.parameters)>1:

            idxDistribution = self.kdeSettings['annealed_distribution_for_transformation']
            use_sample_weight = sample_weight_all[idxDistribution]
            use_sample_weight = use_sample_weight / np.max(use_sample_weight)

            # if enough samples contribute significantly to distribution, use weighted covariance
            if np.sum(use_sample_weight>0.1) > len(self.parameters)*2:
                eigvec = np.linalg.eig( np.cov( theta.transpose(), aweights = use_sample_weight )  )
            else:
                theta1 = theta[idx,:]
                eigvec = np.linalg.eig( np.cov( theta1.transpose() )  )
            W = eigvec[1]

            # TODO: check if how this influences score
            #W = np.linalg.inv(W * P)
        self.W = W
        self.M = M

        for iTheta in range(theta.shape[0]):
            theta[iTheta,:] = np.dot(np.linalg.inv(self.W),theta[iTheta,:]-self.M)


        self.nPar            = theta.shape[1]
        self.sampler_weight  = [[] for i in range(self.nPar)]
        self.sampler_width   = [[] for i in range(self.nPar)]
        self.sampler_kernels = [[] for i in range(self.nPar)]

        if ('annealed_split_distribution' in self.kdeSettings and
            self.kdeSettings['annealed_split_distribution']):
            swa = sample_weight_all[0]
        else:
            swa = sample_weight_all

        if np.sum(swa)<1e-100:
            swa = swa + 1e-100
        n_eff = np.sum(swa)/np.max(swa)

        ploMax = 1-1/np.max([1.00001, n_eff])
        ploMin = 0.05

        if ploMax < ploMin:
            plo = np.array([ploMin])
        else:
            plo = np.linspace(ploMin, ploMax, 10)

        weight_factor = 1
        if 'annealed_scale_samplers' in self.kdeSettings and self.kdeSettings['annealed_scale_samplers']:
            weight_factor = 1 / np.max([plo])

        # add point to sampler
        for iPar in range(self.nPar):
            thetaPar = theta[:,iPar]

            if ('annealed_split_distribution' in self.kdeSettings and
                self.kdeSettings['annealed_split_distribution']):
                for i in range(len(sample_weight_all)):
                    if self.kdeSettings['annealed_weight_multiple'][i]>0:
                        self.add_kernels(plo, iPar, thetaPar, X2, sample_weight_all[i], weight_factor)
                        #(plo, iPar, thetaPar, X2, sample_weight_all[i] ,
                         #                weight_factor=self.kdeSettings['annealed_weight_multiple'][i] )

            else:
                self.add_kernels(plo, iPar, thetaPar, X2, sample_weight_all)



        self.sampler_weight_cumsum = [np.cumsum(self.sampler_weight[i]) for i in range(self.nPar)]

    def add_kernels(self, plo, iPar, theta, X2, sample_weight, weight_factor=1):
        n_eff = np.sqrt(theta.shape[0]/10)
        nBins = int(np.max([5,np.min([np.sqrt(n_eff), 1000])]))

        plo_weight = np.diff(np.append(0, plo))

        for iPlo in range(len(plo)):

            # determine nbins for this theta
            asTheta = np.argsort(theta)
            cssw = np.cumsum(sample_weight[asTheta])

            t0 = np.min(theta)
            t05 = theta[ asTheta[len(cssw)-(cssw<0.05*cssw[-1])[::-1].argmax() - 1] ]
            t95 = theta[ asTheta[(cssw>0.95*cssw[-1])[::1].argmax()] ]
            t100 = np.max(theta)

            nBins1 = int( nBins * 0.9* (t100-t0) / np.max([0.01*(t100-t0), (t95-t05)]) )

            hi, bi = np.histogram(theta, weights=sample_weight, bins=nBins1)
            thetaMin, thetaOpt, thetaMax, height = histFindMinOptMax(hi, bi, plo[iPlo])
            binwidth = bi[1] - bi[0]
            fullWidth = thetaMax-thetaMin

            # widen search area when thetaopt is too close to the boundaries
            fac_opt = 0.25
            fac_bin = 0
            thetaMax = np.max([thetaMax+fac_bin*binwidth, thetaOpt+fac_opt*fullWidth])
            thetaMin = np.min([thetaMin-fac_bin*binwidth, thetaOpt-fac_opt*fullWidth])

            # tst kernel
            w = plo_weight[iPlo] * weight_factor
            k = kernelTophat(
                    thetaMin,
                    thetaMax
                    )

            # check if kernel exists
            addKernel = True
            for iK in range(len(self.sampler_kernels[iPar])):
                if self.sampler_kernels[iPar][iK].a == k.a and self.sampler_kernels[iPar][iK].b == k.b:
                    self.sampler_weight[iPar][iK]+=w
                    addKernel = False
                    break
            if addKernel:
                self.sampler_kernels[iPar].append(k)
                self.sampler_weight[iPar].append( w  )
                self.sampler_width[iPar].append( k.b-k.a )


    def get_sample(self, transform=True):

        sample = np.ndarray(self.nPar)

        for iPar in range(len(sample)):
            u = np.random.random()
            cumsum_weight = self.sampler_weight_cumsum[iPar]
            sum_weight = cumsum_weight[-1]
            i = np.searchsorted(cumsum_weight, u * sum_weight)

            # get sample
            sample[iPar] = self.sampler_kernels[iPar][i].get_sample()

        if transform:
            return np.dot(self.W, sample) + self.M
        return sample


    def score_sample(self,X):
        defg

    def score_samples(self,theta):
        X = np.array(theta)
        for iTheta in range(X.shape[0]):
            X[iTheta,:] = np.dot(np.linalg.inv(self.W),X[iTheta,:]-self.M)


        scores = np.zeros((X.shape[0], X.shape[1]))



        for iPar in range(scores.shape[1]):
            total_weight = self.sampler_weight_cumsum[iPar][-1]

            for iKernel in range(len(self.sampler_weight[iPar])):
                s = self.sampler_kernels[iPar][iKernel].score_samples(X[:,iPar])

                # - self.sampler_weight_cumsum  for numeric issues
                s = s# - self.max_single_weight

                scores[:, iPar]+= np.exp(s) * self.sampler_weight[iPar][iKernel] / total_weight



        # combine scores
        idx = scores<1e-100
        scores[idx] = -np.Infinity
        scores_transform_correction = -np.log(np.abs(np.linalg.det(self.W)))
        #scores[idx==False] = np.log(scores[idx==False]) + self.max_single_weight + scores_transform_correction
        scores[idx==False] = np.log(scores[idx==False]) + scores_transform_correction

        scores = np.sum(scores, axis=1)

        return scores

    def score_samples_1D(self,X,iPar):
        scores = np.zeros((X.shape[0]))

        total_weight = self.sampler_weight_cumsum[-1]

        for iKernel in range(len(self.sampler_weight)):
            s = self.sampler_kernels[iKernel].score_samples_1D(X, iPar)

            # - self.sampler_weight_cumsum  for numeric issues
            s = s #- self.max_single_weight
            scores+= np.exp(s) * self.sampler_weight[iKernel] / total_weight

        # combine scores
        idx = scores<1e-100
        scores[idx] = -np.Infinity
        scores[idx==False] = np.log(scores[idx==False]) #+ self.max_single_weight
        return scores

    def get_important_points_for_plot_distribution(self, iPar):
        points = []

        for kernel in self.sampler_kernels:
            points = points + kernel.get_important_points_for_plot_distribution(iPar)

        return np.sort(points)












def histFindMinOptMax(hi, bi, fac):
    sum_hist = np.sum(hi)
    if sum_hist<1e-100:
        return np.min(hi), np.mean(hi), np.max(hi), 0

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




class kernelTriangle:
    def __init__(self, a, b, c=None):
        # a = min
        # b = max
        # c = mid

        if c is None:
            c = (a+b)/2

        self.a = a
        self.b = b
        self.c = c

    def get_sample(self):
        X = np.random.random(len(self.a))
        return np.array([uniformToTriangle([X[i]], self.a[i], self.b[i], self.c[i])[0] for i in range(len(self.a))])

    def score_samples(self, X):

        # calculate distribution
        s=np.zeros(X.shape)

        a = self.a
        b = self.b
        c = self.c

        for iPar in range(len(a)):
            # a <= X < c :
            idx = (X[:,iPar] > a[iPar]) & (X[:,iPar] <= c[iPar])
            s[idx,iPar] = 2 * (X[idx,iPar] - a[iPar]) /( (b[iPar] - a[iPar]) * (c[iPar] - a[iPar]))

            # c <= X < b :
            idx = (X[:,iPar] > c[iPar]) & (X[:,iPar] < b[iPar])
            s[idx,iPar] = 2 * (b[iPar] - X[idx,iPar]) /( (b[iPar] - a[iPar]) * (b[iPar] - c[iPar]))

        # map to
        idx = np.any(s<1e-100,axis=1)
        s[idx,:]=-np.Infinity
        s[idx==False,:] = np.log(s[idx==False,:])
        s = np.sum(s, axis=1)

        return s

    def get_important_points_for_plot_distribution(self, iPar):
        return [self.a[iPar]-1e-10, self.a[iPar], self.c[iPar], self.b[iPar], self.b[iPar]+1e-10]



class kernelTophat:
    def __init__(self, a, b):
        self.a = np.array(a)
        self.b = np.array(b)

        self.loglik = np.sum(np.log(1/(self.b-self.a)))

    def get_sample(self):
        X = np.random.random(1)
        return self.a + (self.b-self.a)*X

    def score_samples(self,X):
        s = (X<self.a) | (X>self.b)

        s = s==False
        score = np.zeros(s.shape)-np.Infinity
        score[s] = self.loglik

        return score

    def get_important_points_for_plot_distribution(self, iPar):
        return [self.a[iPar]-1e-10, self.a[iPar], self.b[iPar], self.b[iPar]+1e-10]



def uniformToTriangle(X,a=-1,b=1,c=0):
    # X=point [0,1)
    # a,b,c = min,max,mid
    X=np.array(X)
    Fc = (c-a)/(b-a)
    Y=np.ndarray(X.shape)
    Y[X<Fc]= a + np.sqrt( X[X<Fc] * (b-a) * (c-a) )
    Y[X>Fc]= b - np.sqrt( (1-X[X>Fc]) * (b-a) * (b-c) )
    return Y
