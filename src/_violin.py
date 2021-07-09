
# VIOLIN
import weighted
from matplotlib.cbook import violin_stats
from scipy import stats
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

def violin(loc,x,w, c=None, left=None, right=None, percentile=[], nConvolve = 10):
    v_hist, v_bins = np.histogram(x, weights=w, bins=100)
    
    if left is None:
        left =1
    if right is None:
        right = 1
    
    nConv = 2
    cConv = 3
    
    for _ in range(nConvolve*nConv+1):
        v_hist = np.append(0, v_hist)
        v_bins = np.append(v_bins[0]-np.diff(v_bins[:2]), v_bins)
        v_hist = np.append(v_hist, 0)
        v_bins = np.append(v_bins, v_bins[-1]+np.diff(v_bins[:2]))
        
    f = lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi)
    
    for _ in range(nConvolve):
        v_hist = np.convolve(v_hist, f(np.linspace(-nConv,nConv,nConv*2+1)/cConv), 'same')
    
    # remove 1% lowest density
    area_original = np.sum(v_hist)
    idx1 = 0
    idx2 = -1
    a1=0
    a2=0
    while np.sum(v_hist) > 0.95*area_original and idx1 < len(v_hist) and idx2 < len(v_hist):
        if a1>a2:
            a2=a2+v_hist[idx2]
            v_hist[idx2]=0
            idx2=idx2-1
        else:
            a1=a1+v_hist[idx1]
            v_hist[idx1]=0
            idx1=idx1+1
        
        
    
    idx = v_hist > 0
    v_bins = v_bins[:-1]+np.diff(v_bins[:2])
    v_hist = v_hist[idx]
    v_bins = v_bins[idx]
    

    ax = plt.gca()
    
    v_hist = v_hist / np.max(v_hist)
    
    x1 = np.ones(len(v_hist))*loc + right*v_hist
    x2 = np.ones(len(v_hist))*loc - left*v_hist
    y1 = v_bins
    y2 = v_bins
    
    x=np.concatenate((x1, x2[::-1]))
    y=np.concatenate((y1, y2[::-1]))
    
    if not c is None:
        c1 = np.append(np.array(c), 0.5)
    else:
        c1 = None
    
    ax.fill(x, y, zorder=4, fc=c1, ec=c)  
    
    
    cumsum_v_hist = np.cumsum(v_hist)
    cumsum_v_hist  = cumsum_v_hist / cumsum_v_hist[-1]
    
    
    for per in percentile:

        mx1 = np.interp(per, cumsum_v_hist, x1)
        mx2 = np.interp(per, cumsum_v_hist, x2)
        my1 = np.interp(per, cumsum_v_hist, v_bins)
        my2 = my1
        
        print(' res:' ,mx1, mx2, my1, my2)
        
        plt.plot([mx1, mx2], [my1,my2], 'w', zorder=5)
        
    