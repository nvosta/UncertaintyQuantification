"""
    code to test 2 parameters

    by Nick van Osta
"""


import sys
import matplotlib.pyplot as plt
import numpy as np
import os

sys.path.insert(1, 'src/')
import adaptive_importance_sampling
import ais_postprocessing
import plot_adaptive_importance_sampling as pais
import Model
import Patient
import ParametersManual20
import cost as newCost
import Cost
import kernel
import plot_adaptive_importance_sampling_dashboard as paisd
import _stiffness as ST

import addcopyfighandler
import matplotlib
matplotlib.rcParams['savefig.format'] = 'svg'

#%% simulation settings
simName = 'final_AMIS'
settings={
    'save_folder':'data/'+simName+'/',
    'n_sims': 100,
    'save_prior_every': 10,
    'max_temperature': 10,
    'max_temperature_decrease_factor': 0.1**(1/10),
    'min_temperature': 1, # max = min if min>max
    'reduce_min_temperature_tau': 1, # min temperature 1 after 100 iterations
    'reduce_max_temperature_tau': 33, # not used,  max temperature 1 after 200 iterations
    'reduce_max_after_n_iterations': 100, # not used
    'min_max_error': 1,
    'sim_fac': 10,
    'kernel': 'kernel1'
}

#%% load parameters, cost, model
model = Model.Model('src/CircAdapt.dll')
parameters = ParametersManual20.ParametersManual20_NOVWALL()

model_instance = model.getInstance()
X = parameters.getX(model_instance)

settings['init_lb'] = np.array(X) - 0.5
settings['init_ub'] = np.array(X) + 0.5


#%% Set likelihood function
cost = newCost.cost()
cost.rateconst = [0.02,0.02,0.02,0.02,0.02]
cost.rateconstFull = [0.02,0.02,0.02,0.02,0.02]#[1, 1, 1, 1, 1]

cost.segment_difference_const = 0.25
cost.segment_difference_full_range_const = 0.0

cost.normalize_modelled_strain = True
cost.strainFac = [0.2,0.2,0.2,0.2,0.2] # RV1,2,3, LV, SV
cost.strainFacFull = [0,0,0,0,0] # RV1,2,3, LV, SV
cost.facGRstrain = 0.2

cost.splitRVLV = False
cost.useEDV = True
cost.useLVEF = True
cost.constEDV = 10

cost.useLVdiastolicPositiveStrain = True
cost.useMaxMLAP = True
cost.useMaxMRAP = True
cost.useNonNegativeVolumes = True
cost.useMaxSfAct = True
cost.maxSfAct = 1000000
cost.useMinAVdelay = True
cost.useMaxRVEF = True
cost.useMaxDt = True
cost.useNoBulging = True
cost.useAmRefRatio = True
cost.useRVD = True
cost.constRVD = 2.5
cost.useMitValve = False
cost.useAtrialKick = True
cost.useMaxTimeToPeakstrain = True
cost.usePrestretchPenalty = True

cost.convoluteStrainRate=True

cost.useMink1 = False
cost.useMinSfAct = False
cost.useStrainTime = False

# cost Save indices
costSaveIndices = []
settings['costSaveIndices'] = costSaveIndices

#%% patients

model_instance = model.getInstance()
model_instance.run()

patients=[]

patients=[]
if True:
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_2_F1.npy'))
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_3_F1.npy'))
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_5_F1.npy'))
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_8_F1.npy'))
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_9_F1.npy'))
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_10_F1.npy'))
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_15_F1.npy'))
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_18_F1.npy'))
    patients.append(Patient.Patient(filename='virtual_patients/vII202005_19_F1.npy'))

for iPat in range(len(patients)):
    # remove Teich
    if patients[iPat].id[:len('II202005_5')] == 'II202005_5':
        patients[iPat].scalarData['LVEDV_Echo'] = int(patients[iPat].scalarData['LVEDV_Echo'][:3])
        patients[iPat].scalarData['LVEF_Echo'] = np.mean([int(patients[iPat].scalarData['LVEF_Echo'][:2]), int(patients[iPat].scalarData['LVEF_Echo'][3:5])])
        print(patients[iPat].scalarData)

    if patients[iPat].id[:len('II202005_15')] == 'II202005_15':
        patients[iPat].scalarData['LVEDV_Echo'] = int(patients[iPat].scalarData['LVEDV_Echo'][:3])
        patients[iPat].scalarData['LVEF_Echo'] = int(patients[iPat].scalarData['LVEF_Echo'][:3])
        print(patients[iPat].scalarData)

#%% Start protocol
ais_job = adaptive_importance_sampling.adaptive_importance_sampling_job(
    model,
    parameters,
    cost,
    patients,
    run_parallel=True,
    n_cores=20,
    options=settings)

ais_job.run(500, min_n_eff=0) # 500

#%% save dummy dicts for postprocessing
if True:
    #%%
    for ais in ais_job.ais: # [ais_job.ais[2]]:
    #for ais in [ais_job.ais[0]]:
        #s = kernel.final_sampler(ais)


        redo = False
        filename = ais.options['save_folder']+'../'+'dummy_datadict_'+ais.patient.id+'.npy'
        if not redo and os.path.isfile(filename):
            dummy_datadict = np.load(filename, allow_pickle=True).item()
        else:
            print( 'Load Kernel patient ', ais.patient.id )
            s = kernel.kernelDensity(ais.prior_data['theta_info']['theta'],
                                     ais.prior_data['theta_info']['X2'],
                                     ais.prior_data_merge_q(ais.prior_data))
            print( 'Finished' )

            dummy_datadict = paisd.get_dummy_datadict(ais, s, n_samples=100)
            np.save(filename, dummy_datadict, allow_pickle=True)
