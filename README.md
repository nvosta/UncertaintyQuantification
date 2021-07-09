# UncertaintyQuantification
Source code used in our paper "Uncertainty quantification of regional cardiac tissue properties in Arrhythmogenic Cardiomyopathy using adaptive multiple importance sampling"

Nick van Osta1*, Feddo P Kirkels2*, Tim van Loon1, Tijmen Koopsen1, Aurore Lyon1, Roel Meiburg1, Wouter Huberts1, Maarten J Cramer2, Tammo Delhaas1, Kristina H Haugaa3, Arco J Teske2, Joost Lumens1

*NvO and FK contributed equally.

1. Department of Biomedical Engineering, Cardiovascular Research Institute Maastricht, Maastricht University, Maastricht, The Netherlands.
2. Department of Cardiology, division Heart and Lungs, University Medical Center Utrecht, Utrecht, the Netherlands.
3. Dept of Cardiology, Oslo University Hospital and University of Oslo, Norway

# Abstract

## Introduction

Computational models of the cardiovascular system are widely used to simulate cardiac (dys)function. Personalization of such models for patient-specific simulation of cardiac function remains challenging. Both measurement and model uncertainty affect accuracy of parameter estimations. In this study, we present a methodology for patient-specific estimation and uncertainty quantification of parameters in the closed-loop CircAdapt model of the human heart and circulation using echocardiographic deformation imaging. Based on patient-specific estimated parameters we aim to reveal the mechanical substrate underlying deformation abnormalities in patients with arrhythmogenic cardiomyopathy (AC). 

## Methods 

We used adaptive multiple importance sampling to estimate the posterior distribution of regional myocardial tissue properties. This methodology is implemented in the CircAdapt cardiovascular modelling platform and applied to estimate active and passive tissue properties underlying regional deformation patterns, left ventricular volumes, and right ventricular diameter. First, we validated the accuracy of this method and its inter- and intraobserver variability using nine datasets obtained in AC patients. Second, we validated the trueness using nine in silico generated virtual patient datasets representative for various stages of AC. Finally, we applied this method to two longitudinal series of echocardiograms of two pathogenic mutation carriers without established myocardial disease at baseline. 
## Results
Tissue characteristics of virtual patients were accurately estimated with a highest density interval around the true parameter value of 9% (95% CI [0 – 79]). Estimated uncertainty of both likelihood and posterior distributions in patient data and virtual data were comparable, supporting the reliability of the patient estimations. Estimations were highly reproducible with an overlap in posterior distributions of 89.9% (95% CI [60.1 – 95.9]). Clinically measured deformation, ejection fraction, and end-diastolic volume were accurately simulated. In presence of worsening of deformation over time, estimated tissue properties also revealed functional deterioration.
## Conclusion
This method facilitates patient-specific simulation-based estimation of regional ventricular tissue properties from non-invasive imaging data, taking into account both measurement and model uncertainties. Two proof-of-principle case studies suggested that this cardiac digital twin technology enables quantitative monitoring of AC disease progression in early stages of disease.
