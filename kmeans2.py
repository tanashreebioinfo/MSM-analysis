import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

data = pd.read_csv("al_r1_r2_phi_m180_p180.dat",header=None,delim_whitespace=True)
data = np.array(data)

npdata = np.array(data)
npdata_filtered = npdata

X_scaled = preprocessing.normalize(npdata_filtered)
kmeans = KMeans(n_clusters=4, random_state=0, n_jobs=-1).fit(npdata_filtered)

classes = {}
mx = -1
for x in kmeans.labels_:
    mx = max(mx, x)
print(mx)

#print(list(zip(kmeans.labels_, npdata_filtered)))

for i,j in zip(kmeans.labels_, npdata_filtered):
        if i not in classes.keys():
            classes[i] = [j]
        else:
            classes[i].append(j)

print len(classes[0])/(677884)
print len(classes[1])/(677884)
print len(classes[2])/(677884)
print len(classes[3])/(677884)

#print(classes[0])
r1 = []
r2 = []
theta= []
colors = []
for color, points in classes.items():
    for point in points:
        colors.append(color)
        r1.append(point[0])
        r2.append(point[1])
        theta.append(point[2])

#print(r1[:10])
#print(r2[:10])
#print(theta[:10])

#get_ipython().magic(u'matplotlib')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(r1, r2, theta, c=colors,s=0.5)

fig.show()

fig.savefig('./kmeans_1.png')   # save the figure to file
#plt.close(fig)

