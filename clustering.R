#clustering using mclust
library(mclust)
data<-read.table("tser_r1_r2_phi_si.dat")
data1<-data.frame(data$V2,data$V3,data$V4)
fit <- Mclust(data1)
plot(fit)


#3D-Kmeans
library(reshape2)   # for melt(...)
library(rgl)        # for plot3d(...)

set.seed(1)         # to create reproducible sample

# 3D matrix, values clustered around -2 and +2
m      <- c(rnorm(500,-2),rnorm(500,+2))
dim(m) <- c(10,10,10) 
v      <- melt(m, varnames=c("x","y","z"))  # 4 columns: x, y, z, value
# interactive 3D plot, coloring based on value
plot3d(v$x,v$y,v$z, col=1+round(v$value-min(v$value)),size=5)
# identify clusters
v      <- scale(v)                          # need to scale or clustering will fail
v      <- data.frame(v)                     # need data frame for later
d  <- dist(v)                               # distance matrix
km <- kmeans(d,centers=2)                   # kmeans clustering, 2 clusters
v$clust <- km$cluster                       # identify clusters
# plot the clusters
plot(z[1:4],col=v$clust)                    # scatterplot matrix
plot3d(v$x,v$y,v$z, col=v$clust,size=5)  


cl <-kmeans(data1[,1:3],3)
plot3d(data1[,1:3], col=cl$cluster, main="k-means clusters")


o=order(cl$cluster)
data.frame(data$V1[o],cl$cluster[o])  # which timeseries belong to which group/class of kmeans

###############clustering and MSM####################
#####################################################
library(msm)
data<-read.table("tser_r1_r2_phi_si.dat")
data<-head(data,21)
colnames(data) <- c("time","r1","r2","theta")
data2<-data.frame(data$r1,data$r2,data$theta,km$cluster)
Q<-statetable.msm(data2$km.cluster, head(data$v1,21), data=data2)

t<-cav.msm <- msm( state ~ years, subject=PTNUM, data = cav,
+                     qmatrix = Q, deathexact = 4)

min_max_normalize <- function(x) 
{
 return ((x - min(x)) / (max(x) - min(x))) }



 my_scale <- function(x) 
{
 return (x - mean(x))/sd(x)}

def listsum(list):
    ret=0
    for i in list:
        ret += i
    return ret

def normlize(data):
	for i in data:
		





