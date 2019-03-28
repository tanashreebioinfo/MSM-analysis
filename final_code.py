#!/usr/bin/env pythoni
from __future__ import print_function
from sklearn.cross_validation import KFold
import scipy.sparse.linalg as sla
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
import matplotlib.pyplot as pp
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

#print (type(kmeanslabel))


lag_times = list(range(1,400,10))
print ("lag", lag_times)
n_timescales = 10
msm_timescales = implied_timescales(kmeanslabel, lag_times, n_timescales=n_timescales, msm=MarkovStateModel(verbose=False))
#print(msm_timescales[:,0])
#for i in range(n_timescales):
#   		print (i)
pp.plot(lag_times, msm_timescales[:,0], 'o-')
pp.plot(lag_times, msm_timescales[:,1], 'o-')
pp.plot(lag_times, msm_timescales[:,2], 'o-')
pp.title('Discrete-time MSM Relaxation Timescales')
pp.semilogy()
pp.show()



#ctmsm_timescales = implied_timescales(kmeanslabel, lag_times, n_timescales=n_timescales, msm=ContinuousTimeMSM(verbose=False))


#X_scaled =  preprocessing.normalize(npdata_filtered)


#sequences2=list(np.transpose(np.reshape(npdata_filtered2[:,3].astype(int),(-1,1))))





######K-Means-Clustering#####
#############################
cluster = KCenters(metric='euclidean', n_clusters=4)

#sequences = cluster.fit_transform(seq)
#for item in sequences:
#	print (item)

'''
kmeans = KMeans(n_clusters=4,random_state=0).fit_transform(npdata_filtered)   #states from kmeans
kmeanslabel=kmeans.labels_.tolist()


########Time scale calculations
lag_times=list(range(1, 100,2))
n_timescales=10

msm_timescales = implied_timescales(sequences, lag_times, n_timescales=n_timescales, msm=MarkovStateModel(verbose=False))

for i in range(n_timescales):
   plt.plot(lag_times, msm_timescales[:, i], 'o-')

plt.title('Discrete-time MSM Relaxation Timescales')
plt.xlabel('lag time')
plt.ylabel('Relaxation Time')
plt.semilogy()
plt.show()


ctmsm_timescales = implied_timescales(sequences, lag_times, n_timescales=n_timescales, msm=ContinuousTimeMSM(verbose=False))
for i in range(n_timescales):
   plt.plot(lag_times, ctmsm_timescales[:, i], 'o-')

plt.title('Continuous-Time MSM Relaxation Timescales')
plt.xlabel('lag time')
plt.ylabel('Relaxation Time')
plt.semilogy()
plt.show()

####Net Flux / Rate/cktest using pyemma#####
cont_time_msm = ContinuousTimeMSM(lag_time=10, n_timescales=None)
time_model=cont_time_msm.fit(sequences)
time_model.ratemat_
N_flux=tpt.fluxes(1,3,time_model,for_committors=None)
MFPTS=tpt.mfpts(time_model)
MLMSM_good = msm.estimate_markov_model(sequences, 4)
ck_good_msm=MLMSM_good.cktest(2)
mplt.plot_cktest(ck_good_msm)



fig, axes = mplt.plot_cktest(your_cktest_object)
fig.savefig('my_file.png', dpi=300)  


NetworkPlot.plot_network(P)   #TPT Network ploting
#accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))  ## acuuracy of the Kmenas 
#print kmeanslabel
msmana.statdist(P)# Stationary destribution


#######MSM modellin with lag time=10################
mle_msm = MarkovStateModel(lag_time=10)        
model=mle_msm.fit(kmeanslabel)
trans_mat2=mle_msm.transmat_  
print (trans_mat2)      
np.savetxt('data2.csv', trans_mat2, delimiter=',')  
eval, evec = sla.eigs(trans_mat2.T, k=1, which='LM')  
#w, v = numpy.linalg.eig(trans_mat1)    #eigenvalues/eigenvectors


#########Continuous-time Markov stateModel##########
cont_time_msm = ContinuousTimeMSM(lag_time=10, n_timescales=None)
time_model=cont_time_msm.fit(kmeanslabel)
N_flux=tpt.net.fluxes(0,3,model,for_committors=None)

#####MiniBatchKMeans-Clustering######
#####################################

rs = np.random.RandomState(42)
#print rs
clusterer = MiniBatchKMeans(n_clusters=4, random_state=rs)
clustered_data = clusterer.fit_transform(msmdata)
msm = MarkovStateModel(lag_time=2)
assignments = msm.fit_transform(clustered_data)
#print assignments
data = np.concatenate(msmdata, axis=0)

#####Ploting network from transition matrix######
################################################

input_data = pd.read_csv('data.csv', index_col=0)
G = nx.DiGraph(input_data.values)
nx.draw(G)

plt.tight_layout()
plt.show()
plt.savefig("Graph.png", format="PNG")



#####cross-validation and the generalized matrix Rayleigh quotient (GMRQ) for selecting MSM hyperparameters######
#################################################################################################################

def fit_and_score(trajectories, model, n_states):
    cv = KFold(len(trajectories), n_folds=2)
    results = []

    for n in n_states:
        model.set_params(grid__n_bins_per_feature=n)
        for fold, (train_index, test_index) in enumerate(cv):
            train_data = [trajectories[i] for i in train_index]
            test_data = [trajectories[i] for i in test_index]

            # fit model with a subset of the data (training data).
            # then we'll score it on both this training data (which
            # will give an overly-rosy picture of its performance)
            # and on the test data.
            model.fit(train_data)
            train_score = model.score(train_data)
            test_score = model.score(test_data)

            results.append({
                'train_score': train_score,
                'test_score': test_score,
                'n_states': n,
                'fold': fold})
    return results

results = fit_and_score(msmdata, model, [5, 10, 25, 50, 100, 200, 500, 750])
print (results)


class1 = []
class2 = []
for i,j in zip(kmeans.labels_,npdata_filtered):
    if i==1:
        class1.append(j) # clusterwise data
    else :
        class2.append(j)

'''
