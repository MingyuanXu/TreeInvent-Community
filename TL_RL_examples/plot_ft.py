import numpy as np
import matplotlib.pyplot as plt
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from scipy.stats import norm
plt.rc('font',size=12)
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

def set_ax_frame(ax):
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    return 
steps=[]
lrs=[]
losses=[]
scores=[]
valid_scores=[]
validities=[]
uniquenesses=[]

with open ('./MFTs_Gen_Model/Training.log','r') as f:
    for line in f.readlines():
        if line[:4]=='Step': 
            var=line.strip().split()
            step=int(var[0][5:].split('/')[0])
            score=float(var[-6])
            valid_score=float(var[-3])
            validity=float(var[-1])
            print (step,score,valid_score,validity)
            steps.append(step)
            scores.append(score)
            valid_scores.append(valid_score)
            validities.append(validity)
            

figure=plt.figure(figsize=(12,6))
gs=gridspec.GridSpec(1,1)   
ax=plt.subplot(gs[0,0])
plt.plot(steps,scores,label='Average Tanimoto Similarity',color='red',marker='o',linewidth=2,markersize=8)
plt.plot(steps,valid_scores,label='Average Tanimoto Similarity of Valid molecules',color='blue',marker='o',linewidth=2,markersize=8)
plt.xlim(0,200)
leg=plt.legend(fancybox=True,framealpha=0,fontsize=12,markerscale=0.5)
plt.tick_params(length=5,top=True,bottom=True,left=True,right=True)
set_ax_frame(ax)
"""
ax=plt.subplot(gs[1,0])
plt.plot(steps,validities,label='validity',color='red',marker='o',linewidth=2,markersize=8)
plt.xlim(0,200)
leg=plt.legend(fancybox=True,framealpha=0,fontsize=12,markerscale=0.5)
plt.tick_params(length=5,top=True,bottom=True,left=True,right=True)
set_ax_frame(ax)
"""
plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.savefig('Transfer_learning.png')
plt.show()

