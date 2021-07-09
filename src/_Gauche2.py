# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import _stiffness as ST
from matplotlib.patches import Circle, Wedge, Rectangle


####################################################################
# GAUGE 

def degree_range(n, gaugeAngle = 270, gaugeStart = -45): 
    start = np.linspace(gaugeStart,gaugeStart+gaugeAngle,n, endpoint=True)[0:-1]
    end = np.linspace(gaugeStart,gaugeStart+gaugeAngle,n, endpoint=True)[1::]
    tick_points = np.append(start, end[-1])# start + ((end-start)/2.)
    return np.c_[start, end], tick_points

def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation

def gauge(ax, labels=['0%','25%','50%','75%','100%'], \
          colors='jet_r', arrow=np.array([1]), title='',arrowColor=[], logscale=False, 
          uncertainty_left=None, uncertainty_right=None, uncertainty_left1=None, uncertainty_right1=None):

    """
    some sanity checks first

    """

    N = len(labels)

    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors
    """

    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list):
        if len(colors) == N:
            colors = colors[::-1]
        else:
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))

    """
    begins the plotting
    """
    gaugeAngle = 240# 270
    gaugeStart = -30#-45
    
    gaugeOuter = 0.5
    gaugeWidth = 0.2
    
    ang_range, tick_points = degree_range(N, gaugeAngle=gaugeAngle, gaugeStart=gaugeStart)
    print(tick_points)

    labels = labels[::-1]

    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors):
        # sectors
        #patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        #patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor="w", edgecolor="k", lw=2))
        pass
    
    patches.append(Wedge((0.,0.), gaugeOuter, np.min(ang_range), np.max(ang_range), width=gaugeWidth, facecolor=[0,0,0,0], edgecolor="k", lw=2, zorder=99))
    [ax.add_patch(p) for p in patches]




    """
    plots the arrow now
    """

    

    pos = gaugeAngle+gaugeStart-arrow*gaugeAngle # mid_points[abs(arrow - N)]
    uncertainty_left = gaugeAngle+gaugeStart-np.array(uncertainty_left) * gaugeAngle
    uncertainty_right = gaugeAngle+gaugeStart-np.array(uncertainty_right) * gaugeAngle
    uncertainty_left1 = gaugeAngle+gaugeStart-np.array(uncertainty_left1) * gaugeAngle
    uncertainty_right1 = gaugeAngle+gaugeStart-np.array(uncertainty_right1) * gaugeAngle
    
    lim = -gaugeStart
    pos[pos<-lim] = -lim
    uncertainty_left[uncertainty_left<-lim] = -lim
    uncertainty_right[uncertainty_right<-lim] = -lim
    uncertainty_left1[uncertainty_left1<-lim] = -lim
    uncertainty_right1[uncertainty_right1<-lim] = -lim
    
    pos[pos>180+lim] = 180+lim
    uncertainty_left[uncertainty_left>180+lim] = 180+lim
    uncertainty_right[uncertainty_right>180+lim] = 180+lim
    uncertainty_left1[uncertainty_left1>180+lim] = 180+lim
    uncertainty_right1[uncertainty_right1>180+lim] = 180+lim
        
    

    patches=[] # for uncertainty
    patches1=[] # for uncertainty
    for ip in range(len(pos)):
        fc = 'k'
        ec = 'k'
        if len(arrowColor)>0:
            fc = arrowColor[ip]
        if uncertainty_left is not None:
            theta0 = uncertainty_right[ip]
            theta1 = uncertainty_left[ip]
            theta01 = uncertainty_right1[ip]
            theta11 = uncertainty_left1[ip]
            
            w = gaugeWidth
            
            nStripe = 1
            w1 = w / nStripe
            w2 = w1 / len(pos) 
            for iStripe in range(nStripe):
                patches1.append(Wedge((0.,0.), gaugeOuter-(ip)*w2- iStripe*w1, theta0, theta1, width=w2, edgecolor=fc, facecolor=[1,1,1], lw=1, zorder=2, alpha=1))
                patches1.append(Wedge((0.,0.), gaugeOuter-(ip)*w2- iStripe*w1, theta0, theta1, width=w2, facecolor=fc, lw=2, zorder=2, alpha=0.5))
                patches1.append(Wedge((0.,0.), gaugeOuter-(ip)*w2- iStripe*w1, theta01, theta11, width=w2, facecolor=fc, lw=2, zorder=2))
                
            # lines
            #for theta in [theta0, theta1]:
            #    tickX1, tickY1 = gaugeOuter * np.cos(np.radians(theta)), gaugeOuter * np.sin(np.radians(theta))
            #    tickX2, tickY2 = (gaugeOuter-gaugeWidth)* np.cos(np.radians(theta)), (gaugeOuter-gaugeWidth) * np.sin(np.radians(theta))
                #plt.plot([tickX1, tickX2], [tickY1, tickY2], c=fc, linewidth=1, zorder=5)
            
    [ax.add_patch(p) for p in patches] # add uncertainty
    [ax.add_patch(p) for p in patches1] # add uncertainty
    
    for ip in range(len(pos)):
        fc = 'k'
        ec = 'k'
        if len(arrowColor)>0:
            fc = arrowColor[ip]
        headLength = 0.1
        arrowLen = 0.3-headLength
        arrowWidth = 0.05
        
        ax.arrow(0, 0, arrowLen * np.cos(np.radians(pos[ip])), arrowLen * np.sin(np.radians(pos[ip])), \
                     width=arrowWidth, head_width=0.1, head_length=headLength, fc=fc, ec=ec,zorder=9997)
            

    

    ax.add_patch(Circle((0, 0), radius=arrowWidth/2, facecolor='w', edgecolor='k',zorder=9998))
    #ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=9999))
    
    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    """
    
    if logscale:
        nSmallTicks = 9
    else:
        nSmallTicks = 10
    
    for iT in range(len(tick_points)-1):
        for st in range(nSmallTicks):# np.linspace(gaugeStart, gaugeStart+gaugeAngle,nSmallTicks):
            if logscale:
                st1 = gaugeStart + (gaugeAngle / (len(tick_points)-1)) * iT
                st1+= np.log(st+1)/np.log(10) * (gaugeAngle / (len(tick_points)-1))
            else:
                st1 = gaugeStart + (gaugeAngle / (len(tick_points)-1)) * iT
                st1+= st / nSmallTicks * (gaugeAngle / (len(tick_points)-1))
                
            
            ticloc = [[gaugeOuter, gaugeOuter-0.02]]
            for tl in ticloc:
                tickX1, tickY1 = tl[1] * np.cos(np.radians(st1)), tl[1] * np.sin(np.radians(st1))
                tickX2, tickY2 = tl[0] * np.cos(np.radians(st1)), tl[0] * np.sin(np.radians(st1))
                plt.plot([tickX1, tickX2], [tickY1, tickY2], c=[0,0,0], linewidth=1, zorder=99)
        
    

    for mid, lab in zip(tick_points, labels):
        ticloc = [[gaugeOuter-gaugeWidth, gaugeOuter-gaugeWidth+0.03],[gaugeOuter, gaugeOuter-0.03]]
        for tl in ticloc:
            tickX1, tickY1 = tl[1] * np.cos(np.radians(mid)), tl[1] * np.sin(np.radians(mid))
            tickX2, tickY2 = tl[0] * np.cos(np.radians(mid)), tl[0] * np.sin(np.radians(mid))
            plt.plot([tickX1, tickX2], [tickY1, tickY2], c=[0,0,0], linewidth=2, zorder=999)
        w = gaugeOuter*1.15
        ax.text(w * np.cos(np.radians(mid)), w * np.sin(np.radians(mid)), str(lab), \
            horizontalalignment='center', verticalalignment='center', fontsize=14, 
            rotation = rot_text(mid))

    
    ax.text(0, gaugeOuter*1.4, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=12, fontweight='bold', zorder=999999999)

    """
    removes frame and ticks, and makes axis equal and tight
    """

    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')

    #yl = plt.ylim()
    #ax.set_ylim([0,yl[1]-yl[0]])
    #ax.set_ylim([0,0.3])
    


####################################################################
# end Gauge

if __name__ == "__main__":
    plt.figure(1)
    plt.clf()
    RV_fill_color = [[135/255, 195/255, 91/255],
                     [34/255, 113/255, 181/255],
                     [236/255, 45/255, 36/255],]
    gauge(plt.gca(),
          arrow=np.array([0.25, 0.4, 0.5]), 
          title='Title', 
          arrowColor=RV_fill_color, 
          logscale=True, 
          uncertainty_left=[0.2, 0.25, 0.46], 
          uncertainty_right=[0.45, 0.5, 0.6])

    