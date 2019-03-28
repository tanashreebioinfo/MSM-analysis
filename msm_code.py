#!/usr/bin/env pythoni
from __future__ import print_function
from sklearn.cross_validation import KFold
import scipy.sparse.linalg as sla
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import msmtools.analysis as msmana
#import msmexplorer as msme
from mdtraj.utils import timing
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.cluster import NDGrid
from msmbuilder.msm import BayesianMarkovStateModel, MarkovStateModel , ContinuousTimeMSM
from sklearn.cluster import KMeans
from msmbuilder.cluster import KCenters
from sklearn.metrics import accuracy_score
#import networkx as nx
from sklearn import preprocessing
from msmbuilder.msm import implied_timescales
from msmbuilder import tpt
import pyemma.plots as mplt
import pyemma.msm as msm
import pyemma.plots as NetworkPlot

####data import/processing######
################################

data = pd.read_csv("clust2.5_si.dat",header=None,delim_whitespace=True)
npdata = np.array(data)
npdata_filtered = npdata[:,4:]
sequences=[]
sequences.append(npdata_filtered)

kmeanslabel=list(npdata[:,4].astype(int))

lag_times=list(range(1, 100,2))
n_timescales=10

msm_timescales = implied_timescales(kmeanslabel, lag_times, n_timescales=n_timescales, msm=MarkovStateModel(verbose=False))
#for i in range(n_timescales):
#   plt.plot(lag_times, msm_timescales[:, i], 'o-')

#plt.title('Discrete-time MSM Relaxation Timescales')
                                                                           
#plt.xlabel('lag time')
#plt.ylabel('Relaxation Time')
#plt.semilogy()
#plt.show()


cont_time_msm = ContinuousTimeMSM(lag_time=10, n_timescales=None)
time_model=cont_time_msm.fit(kmeanslabel)
print (time_model.ratemat_)
P=time_model.transmat_
M = msm.markov_model(P)
NetworkPlot.plot_network(P)

#mplt.plot_markov_model(M);

MLMSM_good = msm.estimate_markov_model(kmeanslabel, 4)
ck_good_msm=MLMSM_good.cktest(4)
fig, axes = mplt.plot_cktest(ck_good_msm)
fig.savefig('my_file.png', dpi=300)

