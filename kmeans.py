import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

data = pd.read_csv("tser_r1_r2_phi_si.dat",header=None,delim_whitespace=True)
data = np.array(data)

npdata = np.array(data)
npdata_filtered = npdata[:,1:]

#kmeans = KMeans(n_clusters=2, random_state=0).fit(npdata_filtered)
X_scaled = preprocessing.scale(npdata_filtered)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_scaled)

class1 = []
class2 = []
for i,j in zip(kmeans.labels_,X_scaled):
    if i==1:
        class1.append(j) 
    else :
	class2.append(j)

#get_ipython().magic(u'matplotlib')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in class1[:1000]:
	#print i[1]
    ax.scatter(i[0],i[1],i[2],c = 'r',s=0.5)
for j  in class2[:1000]:
        #print i[1]
    ax.scatter(j[0],j[1],j[2],c = 'b',s=0.5)

fig.show()
fig.savefig('./kmeans_2.png')   # save the figure to file





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in classes[0]:
    ax.scatter(i[0],i[1],i[2],c = 'r',s=0.5)
for j in classes[1]:
    ax.scatter(j[0],j[1],j[2],c = 'b',s=0.5)
for k in classes[2]:
    ax.scatter(j[0],j[1],j[2],c = 'g',s=0.5)
for l in classes[3]:
    ax.scatter(j[0],j[1],j[2],c = 'y',s=0.5)
fig.show()
fig.savefig('./kmeans_2.png')   # save the figure to file








'''
db=DBSCAN(eps=-1.3, min_samples=10).fit(npdata_filtered)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print (labels)
'''

