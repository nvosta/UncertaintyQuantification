"""
    code to test 2 parameters

    by Nick van Osta
"""

import numpy as np
import os
import kernel
import copy
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

class adaptive_importance_sampling_job:
    def __init__(self,
                 model,
                 parameters,
                 cost,
                 patients,
                 n_cores=2,
                 run_parallel=False,
                 options={}):
        self.options=options

        self.run_parallel = run_parallel
        self.n_cores = n_cores

        # main properties
        self.model = model
        self.parameters = parameters
        self.cost = cost
        self.patients = patients

        # initialize settings
        self.init_folders()

        # init patients
        self.ais = []
        for i_pat in range(len(self.patients)):
            settings_pat = copy.deepcopy(self.options)
            settings_pat['save_folder'] = settings_pat['save_folder'] +  str(patients[i_pat].id) +'/'
            self.ais.append(adaptive_importance_sampling(model,
                                                         parameters,
                                                         cost,
                                                         patients[i_pat],
                                                         settings_pat))

    def init_folders(self):
        """Init all folders."""
        if not os.path.isdir(self.options['save_folder']):
            os.mkdir(self.options['save_folder'])

    def run(self,n_iter, min_n_eff=1):
        if self.run_parallel:
            #for i_ais in range(len(self.ais)):
            #    self.run_single_patient(i_ais, 1)
            print('start parallel')
            Parallel(n_jobs=self.n_cores)(
                self.run_single_patient_parallel(i_ais, n_iter, min_n_eff) for i_ais in range(len(self.ais))
                )
            print('parallel finished')
        # also do this when parallel to load data to main memory space
        for i_ais in range(len(self.ais)):
            print('run set ', i_ais, ', patient dataset ', self.ais[i_ais].patient.id)
            self.run_single_patient(i_ais, n_iter, min_n_eff)

    @delayed
    @wrap_non_picklable_objects
    def run_single_patient_parallel(self, i_ais, n_iter, min_n_eff=1):
        try:
            return self.run_single_patient(i_ais, n_iter, min_n_eff)
        except:
            print('Error in patient ', self.ais[i_ais].patient.id)
        return False


    def run_single_patient(self, i_ais, n_iter, min_n_eff=1):
        self.ais[i_ais].run(n_iter, min_n_eff)
        return True


class adaptive_importance_sampling:
    def __init__(self,
                 model,
                 parameters,
                 cost,
                 patient,
                 options={}):
        self.options = {
            'save_folder':'',
            'n_sims': 10,
            'save_prior_every':1}
        self.samplers = []

        self.options.update(options)

        # main properties
        self.model = model
        self.parameters = parameters
        self.cost = cost
        self.patient = patient

        # initialize settings
        self.init_folders()

    def init_folders(self):
        """Init all folders."""
        if not os.path.isdir(self.options['save_folder']):
            os.mkdir(self.options['save_folder'])

    def run(self, n_iter, min_n_eff=1):
        prior_data = {'cur_iter': 0,
                      'iter_info': {
                          'i_iter':[],
                          'n_sims':[],
                          'temperature':[],
                          'max_temperature':[],
                          'X2min': [],
                          'max_q': []},
                      'theta_info': {}}


        filename = ''
        start_iter = 0 # load prior data at iteration
        for i_iter in range(int(np.max([1e4, n_iter]))):
            if os.path.isfile(self.get_prior_data_filename(i_iter)):
                filename = self.get_prior_data_filename(i_iter)
                start_iter = i_iter+1
        if start_iter>0:
            try:
                prior_data = np.load(filename, allow_pickle=True).item()
            except:
                os.remove(filename)
                print('File '+filename+' not existing or corrupted')
                raise Exception('File '+filename+' not existing or corrupted')


        i_iter = 0


        while i_iter < start_iter:
            q = self.get_sampler(i_iter, {})
            i_iter+=1

        n_eff = 0
        if i_iter>50:
            X2 = prior_data['theta_info']['X2']
            q = self.prior_data_merge_q(prior_data)
            w = np.exp(-X2-q)
            n_eff = np.sum(w)**2 / np.sum(w**2)

        # Check if final exists and data is removed
        if i_iter==0 and os.path.isfile(self.get_prior_data_filename('final')):
            try:
                self.prior_data = np.load(self.get_prior_data_filename('final'), allow_pickle=True).item()
            except:
                os.remove(filename)
                print('File '+filename+' not existing or corrupted')
                raise Exception('File '+filename+' not existing or corrupted')

            return

        n_iters_calculated = 0
        while i_iter < n_iter or n_eff < min_n_eff:
            n_iters_calculated = n_iters_calculated + 1

            PdictRef = prior_data['theta_info']['PdictOpt'] if i_iter>0 else None

            # get results
            results = self.run_simulations(i_iter, prior_data,
                                           enforce_success=i_iter==0,
                                           first_iteration=i_iter==0,
                                           PdictRef=PdictRef)

            # get Prior Data
            prior_data = self.merge_results_into_prior_data(i_iter, prior_data, results)

            # next iteration
            i_iter = i_iter + 1

            X2 = prior_data['theta_info']['X2']
            q = self.prior_data_merge_q(prior_data)
            w = np.exp(-X2-q)
            n_eff = np.sum(w)**2 / np.sum(w**2)


        self.prior_data = prior_data
        if n_iters_calculated>0:
            self.save_prior_data(i_iter-1, prior_data)
        if n_iters_calculated>0 or os.path.isfile(self.get_prior_data_filename('final'))==False:
            np.save(self.get_prior_data_filename('final'), prior_data, allow_pickle=True)



    # samplers
    def get_sampler(self, i_iter, prior_data=None):
        filename = self.options['save_folder'] + 'sampler{:05d}.npy'.format(i_iter)
        if len(self.samplers)>i_iter:
            return self.samplers[i_iter]
        elif os.path.isfile(filename):
            try:
                sampler = np.load(filename, allow_pickle=True).item()['sampler']
            except:
                os.remove(filename)
                print('File '+filename+' not existing or corrupted')
                raise Exception('File '+filename+' not existing or corrupted')

        else: # create
            if i_iter==0:
                sampler = kernel.kernel(n_par=len(self.parameters))
                sampler.w_k1 = [1,0,0]
                for i_par in range(len(self.parameters)):
                    sampler.add_tophat(i_par, self.options['init_lb'][i_par], self.options['init_ub'][i_par], 1, iW = 0)
            else:
                Tmin = np.max([1, self.options['min_temperature'] * np.exp(-i_iter/self.options['reduce_min_temperature_tau']) ])
                #Tmax = 1 + self.options['max_temperature'] * np.exp(
                #    -np.max([0, i_iter - self.options['reduce_max_after_n_iterations']])/self.options['reduce_max_temperature_tau'])

                Tmax = prior_data['iter_info']['max_temperature'][-1]


                if ~np.isinf(Tmax):
                    Tmin = np.max([Tmin, Tmax*self.options['max_temperature_decrease_factor']])
                sim_fac = self.options['sim_fac']


                #if np.min(prior_data['theta_info']['X2']) > 50:
                #    Tmin = np.max([Tmin, np.min(prior_data['theta_info']['X2'])/2])
                #    Tmax = np.max([Tmin, Tmax])

                if 'kernel' in self.options:
                    if self.options['kernel']=='kmeans':
                        sampler = kernel.kernelKMeans(
                                prior_data['theta_info']['theta'],
                                prior_data['theta_info']['X2'],
                                self.prior_data_merge_q(prior_data),
                                Tmin=Tmin,
                                Tmax=Tmax,
                                sim_fac=sim_fac,
                                logcorrection= self.parameters.get_log_correction(prior_data['theta_info']['theta'])
                            )
                    elif self.options['kernel']=='kmeans1':
                        sampler = kernel.kernelKMeans1(
                                prior_data['theta_info']['theta'],
                                prior_data['theta_info']['X2'],
                                self.prior_data_merge_q(prior_data),
                                Tmin=Tmin,
                                Tmax=Tmax,
                                sim_fac=sim_fac,
                                logcorrection= self.parameters.get_log_correction(prior_data['theta_info']['theta'])
                            )
                    elif self.options['kernel']=='kernel1':
                        sampler = kernel.kernel1(
                                prior_data['theta_info']['theta'],
                                prior_data['theta_info']['X2'],
                                self.prior_data_merge_q(prior_data),
                                Tmin=Tmin,
                                Tmax=Tmax,
                                logcorrection= self.parameters.get_log_correction(prior_data['theta_info']['theta'])
                            )
                    elif self.options['kernel']=='kernel1a':
                        sampler = kernel.kernel1a(
                                prior_data['theta_info']['theta'],
                                prior_data['theta_info']['X2'],
                                self.prior_data_merge_q(prior_data),
                                Tmin=Tmin,
                                Tmax=Tmax,
                                logcorrection= self.parameters.get_log_correction(prior_data['theta_info']['theta'])
                            )
                    elif self.options['kernel']=='kernel1b':
                        sampler = kernel.kernel1b(
                                prior_data['theta_info']['theta'],
                                prior_data['theta_info']['X2'],
                                self.prior_data_merge_q(prior_data),
                                Tmin=Tmin,
                                Tmax=Tmax,
                                logcorrection= self.parameters.get_log_correction(prior_data['theta_info']['theta'])
                            )
                    elif self.options['kernel']=='kernel1c':
                        sampler = kernel.kernel1c(
                                prior_data['theta_info']['theta'],
                                prior_data['theta_info']['X2'],
                                self.prior_data_merge_q(prior_data),
                                Tmin=Tmin,
                                Tmax=Tmax,
                                logcorrection= self.parameters.get_log_correction(prior_data['theta_info']['theta'])
                            )
                    else:
                        sampler = kernel.kernelSmall(
                                prior_data['theta_info']['theta'],
                                prior_data['theta_info']['X2'],
                                self.prior_data_merge_q(prior_data),
                                Tmin=Tmin,
                                Tmax=Tmax,
                                sim_fac=sim_fac,
                                logcorrection= self.parameters.get_log_correction(prior_data['theta_info']['theta'])
                            )
                else:
                    sampler = kernel.kernel(
                            prior_data['theta_info']['theta'],
                            prior_data['theta_info']['X2'],
                            self.prior_data_merge_q(prior_data),
                            Tmin=Tmin,
                            Tmax=Tmax,
                            sim_fac=sim_fac,
                            logcorrection= self.parameters.get_log_correction(prior_data['theta_info']['theta'])
                        )

            # save
            s = {'sampler': sampler}
            np.save(filename, s, allow_pickle=True)

        self.samplers.append(sampler)
        return sampler

    def prior_data_merge_q(self, prior_data):
        q = np.exp(prior_data['theta_info']['q'])
        q = np.mean(q, axis=1)
        q = np.log(q)
        return q

    # run simulations
    def run_simulations(self, i_iter, prior_data, enforce_success=False, first_iteration = False, PdictRef=None):
        # always load sampler
        q = self.get_sampler(i_iter, prior_data)

        # get results
        filename = self.options['save_folder'] + 'results{:05d}.npy'.format(i_iter)
        if os.path.isfile(filename):
            try:
                results = np.load(filename, allow_pickle=True).item()
            except:
                os.remove(filename)
                print('File '+filename+' not existing or corrupted')
                raise Exception('File '+filename+' not existing or corrupted')

            if not 'temperature' in results:
                results['temperature'] = q.T
            return results

        # make results
        results = {'theta': [], 'X2': [], 'indices': [], 'PdictOpt':[]}
        if first_iteration:
            n_sims = 10*len(self.parameters)+1
        else:
            n_sims = self.options['n_sims']
        i_sim = 0

        # iterate over simulations
        while i_sim < n_sims:
            print(self.patient.id, ' i_sim:', i_sim)

            theta = q.get_sample()

            runFunction = self.options['stabalize_solution'] if 'stabalize_solution' in self.options else 0

            if runFunction==0: # use Popt
            # use only Popt
                for iTry in [0]:#range(1 + int(PdictRef is not None)):
                    try:
                        model_instance = self.model.getInstance()
                        #if iTry==0 and PdictRef is not None and PdictRef != []:
                        #    model_instance.setPdict(PdictRef)
                        model_instance.setScalar('','','','tCycle',np.round(self.patient.RVtime[-1]/0.002)*0.002)
                        theta0 = self.parameters.getX(model_instance)
                        self.parameters.setX(model_instance, 0.5*theta0+0.5*theta)
                        model_instance.runFast()
                        if not np.any(np.isnan(model_instance.getVector('Lv','Cavity','Lv','V'))):
                            self.parameters.setX(model_instance, theta)
                            model_instance.run()
                            if not model_instance.getIsStable() and not np.any(np.isnan(model_instance.getVector('Lv','Cavity','Lv','V'))):
                                model_instance.run()
                        if model_instance.getIsStable():
                            break
                    except:
                        pass
            elif runFunction==1:
                nTry = 2
                model_instance = self.model.getInstance()
                model_instance.setScalar('','','','tCycle',np.round(self.patient.RVtime[-1]/0.002)*0.002)
                theta0 = self.parameters.getX(model_instance)
                for iTry in range(nTry):
                    f = iTry / nTry
                    self.parameters.setX(model_instance, (1-f)*theta0+f*theta)
                    model_instance.runSingleBeat()
                    if np.any(np.isnan(model_instance.getVector('Lv','Cavity','Lv','V'))):
                        break

                if not np.any(np.isnan(model_instance.getVector('Lv','Cavity','Lv','V'))):
                    self.parameters.setX(model_instance, theta)
                    model_instance.run()

            else:
                STOP
                model_instance = self.model.getInstance()
                model_instance.setScalar('','','','tCycle',np.round(self.patient.RVtime[-1]/0.002)*0.002)
                theta0 = self.parameters.getX(model_instance)
                self.parameters.setX(model_instance, 0.5*theta0+0.5*theta)
                model_instance.runFast()
                if model_instance.getIsStable():
                    self.parameters.setX(model_instance, theta)
                    model_instance.run()

            if model_instance.getIsStable() and not np.any(np.isnan(model_instance.getVector('Lv','Cavity','Lv','V'))):
                i_sim += 1
                #try:
                X2 = self.cost.getCost2(self.patient, model_instance)
                if self.options['costSaveIndices']==[]:
                    indices = []
                else:
                    indices = self.options['costSaveIndices'].getCostMod(model_instance=model_instance, patient=self.patient)
                #except:
                #    print('Something went wrong with calculating the cost')

                minX2 = np.Infinity
                if len(results['X2'])>0:
                    minX2 = np.min([minX2, np.min(results['X2'])])
                if not prior_data['theta_info']=={} and len(prior_data['theta_info']['X2'])>0:
                    minX2 = np.min([minX2, np.min(prior_data['theta_info']['X2'])])

                if X2 < minX2:
                    results['PdictOpt'] = model_instance.getPdict()

                results['theta'].append(theta)
                results['X2'].append(X2)
                results['indices'].append(indices)
                print(self.patient.id, ' i_sim:', i_sim, 'success')

            elif not enforce_success:
                i_sim += 1

        results['theta'] = np.array(results['theta'])
        results['X2'] = np.array(results['X2'])
        results['indices'] = np.array(results['indices'])
        results['temperature'] = q.T

        np.save(filename, results, allow_pickle=True)
        return results



    # prior data
    def merge_results_into_prior_data(self, i_iter, prior_data, results):
        # set i_iter
        prior_data['i_iter'] = i_iter

        # get settings
        n_sims = results['theta'].shape[0]

        # append results
        if i_iter==0:
            prior_data['theta_info']=results
            prior_data['theta_info']['q']=np.ndarray((n_sims, 1))
            prior_data['theta_info']['q'][:] = - np.Infinity
            prior_data['theta_info']['q_is_calculated']=np.zeros((n_sims, 1), dtype=bool)
            prior_data['theta_info']['PdictOpt'] = results['PdictOpt']
        elif n_sims>0:
            prior_data['theta_info']['theta']=np.append(prior_data['theta_info']['theta'],
                                                        results['theta'], axis=0)
            X2improvement = np.max([0, np.nanmin(prior_data['theta_info']['X2']) - np.nanmin(results['X2'])])
            if X2improvement > 0:
                prior_data['theta_info']['PdictOpt'] = results['PdictOpt']
            #print('X2improvement: ', X2improvement)
            prior_data['theta_info']['X2']=np.append(prior_data['theta_info']['X2'],
                                                        results['X2'], axis=0)
            prior_data['theta_info']['indices']=np.append(prior_data['theta_info']['indices'],
                                                        results['indices'], axis=0)
            q = prior_data['theta_info']['q']
            q_is_calculated = prior_data['theta_info']['q_is_calculated']

            prior_data['theta_info']['q'] = np.zeros((q.shape[0]+n_sims, q.shape[1]+1)) - np.Infinity
            prior_data['theta_info']['q_is_calculated']=np.zeros((q.shape[0]+n_sims, q.shape[1]+1), dtype=bool)

            prior_data['theta_info']['q'][:q.shape[0], :q.shape[1]] = q
            prior_data['theta_info']['q_is_calculated'][:q_is_calculated.shape[0], :q_is_calculated.shape[1]] = q_is_calculated
        else:
            X2improvement = 0

            q = prior_data['theta_info']['q']
            q_is_calculated = prior_data['theta_info']['q_is_calculated']

            prior_data['theta_info']['q'] = np.zeros((q.shape[0]+n_sims, q.shape[1]+1)) - np.Infinity
            prior_data['theta_info']['q_is_calculated']=np.zeros((q.shape[0]+n_sims, q.shape[1]+1), dtype=bool)

            prior_data['theta_info']['q'][:q.shape[0], :q.shape[1]] = q
            prior_data['theta_info']['q_is_calculated'][:q_is_calculated.shape[0], :q_is_calculated.shape[1]] = q_is_calculated
        # iter_info
        prior_data['iter_info']['i_iter'].append(i_iter)
        prior_data['iter_info']['n_sims'].append(self.options['n_sims']) # tried sims differ from actual sims in result
        prior_data['iter_info']['temperature'].append(results['temperature'])
        prior_data['iter_info']['max_q'] = np.append(prior_data['iter_info']['max_q'], -np.Infinity)
        prior_data['iter_info']['X2min'].append(np.nanmin(prior_data['theta_info']['X2']))
        if i_iter==0:
            prior_data['iter_info']['max_temperature'] = [self.options['max_temperature']]
        else:
            prior_data['iter_info']['max_temperature'].append(
                np.min([
                np.max([1, prior_data['iter_info']['max_temperature'][-1]*self.options['max_temperature_decrease_factor']])+X2improvement*2,
                self.options['max_temperature']
                ])
                )
            if np.isnan(prior_data['iter_info']['max_temperature'][-1]):
                prior_data['iter_info']['max_temperature'][-1] = self.options['max_temperature']
            if prior_data['iter_info']['temperature'][-1] > prior_data['iter_info']['max_temperature'][-1]:
                prior_data['iter_info']['max_temperature'][-1] = prior_data['iter_info']['temperature'][-1]*self.options['max_temperature_decrease_factor']
            if prior_data['iter_info']['temperature'][-1] < 1:
                prior_data['iter_info']['max_temperature'][-1] = 1



        # remove
        prior_data = self.remove_prior_data_unneeded(i_iter, prior_data)


        # calculate proposal probability for all samples
        prior_data = self.calculate_prior_data_q(prior_data)

        # remove data
        # todo

        # store prior_data
        if np.mod(i_iter, self.options['save_prior_every'])==0:
            self.save_prior_data(i_iter, prior_data)

        # return
        return prior_data

    def save_prior_data(self, i_iter, prior_data):
        np.save(self.get_prior_data_filename(i_iter), prior_data, allow_pickle=True)
        for i in range(i_iter-self.options['save_prior_every']*2,i_iter):
            if os.path.isfile(self.get_prior_data_filename(i)):
                os.remove(self.get_prior_data_filename(i))


    def get_prior_data_filename(self, i_iter):
        if i_iter=='final':
            return self.options['save_folder'] + '../prior_data_' + self.patient.id +'_' + i_iter + '.npy'
        return self.options['save_folder'] + 'prior_data{:05d}.npy'.format(i_iter)


    def remove_prior_data_unneeded(self, i_iter, prior_data):
        sampler = self.get_sampler(i_iter, prior_data)

        # idx to remove nan and inf
        idx = np.isnan(prior_data['theta_info']['X2'])
        idx = idx | np.isinf(prior_data['theta_info']['X2'])
        X2min = np.nanmin(prior_data['theta_info']['X2'])

        # remove unimportant samples, not in first iteration
        if len(prior_data['iter_info']['temperature'])>1:
            minTemp = sampler.Ts
            X2lim = X2min + 3*len(self.parameters)*minTemp
            idxBelow = (prior_data['theta_info']['X2'] > X2lim)

            #prevent from empty theta list
            if np.sum(idxBelow==False) > 10*len(self.parameters):
                idx = idx | idxBelow
            elif np.sum(np.isnan(prior_data['theta_info']['X2'])==False)>10*len(self.parameters):
                args = np.argsort(prior_data['theta_info']['X2'])
                idxBelow[args[:10*len(self.parameters)]] = False
                idx = idx | idxBelow



        # idx to keep
        idx = idx==False

        prior_data['theta_info']['theta']=prior_data['theta_info']['theta'][idx,:]
        prior_data['theta_info']['X2']=prior_data['theta_info']['X2'][idx]
        prior_data['theta_info']['q']=prior_data['theta_info']['q'][idx,:]
        prior_data['theta_info']['q_is_calculated']=prior_data['theta_info']['q_is_calculated'][idx,:]
        prior_data['theta_info']['indices']=prior_data['theta_info']['indices'][idx,:]


        return prior_data

    def calculate_prior_data_q(self, prior_data):
        for i_i_iter in range(len(prior_data['iter_info']['i_iter'])):
            idx = prior_data['theta_info']['q_is_calculated'][:,i_i_iter]==False
            if np.sum(idx)>0:
                theta = prior_data['theta_info']['theta'][idx,:]
                q = self.get_sampler(prior_data['iter_info']['i_iter'][i_i_iter])


                # log correction
                #parval = self.parameters.XtoParVal(np.array(theta))
                #logcorrection = np.sum(np.log(parval[:,:10]), axis=1) + np.sum(np.log(parval[:,15:]), axis=1)
                logcorrection = self.parameters.get_log_correction(theta)

                prior_data['theta_info']['q'][idx,i_i_iter] = q.score_samples(theta) + logcorrection

                prior_data['theta_info']['q_is_calculated'][idx,i_i_iter]=True


        prior_data['iter_info']['max_q'] = np.max(
            np.append(prior_data['iter_info']['max_q'].reshape(1,-1), prior_data['theta_info']['q'], axis=0)
            ,axis=0)

        return prior_data
