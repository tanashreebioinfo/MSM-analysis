library(rgl)
data2<-read.table("tser_r1_r2_phi_si.dat")
normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))}
data1<-data.frame(normalize(data2$V2),normalize(data2$V3),normalize(data2$V4))
setEPS()
postscript("data1_nomr_kmeans4.eps")
km <-kmeans(data1,4, iter.max=1000, nstart=20)
plot3d(data1, col=km$cluster, main="k-means clusters")
dev.off()


while (i <= mdim[1] ){
   if (mcol[i] == 1){
      mcol[i] = "red"
   }else {
      mcol[i] = "blue"
   }
   i = i +1
}



km <-kmeans(data1,4, iter.max=10/100, nstart=1, algorithm = "Hartigan-Wong/MacQueen")

#Silicon
awk '{if($2 <=2.4 && $3<=2.4 && $4 <=0 && $4>=-180) {print $0,1}}' tser_r1_r2_phi_si.dat > s1.dat
awk '{if($2 <=2.4 && $3<=2.4 && $4 <=180 && $4>=0) {print $0,2}}' tser_r1_r2_phi_si.dat > s2.dat
awk '{if($2 >=2.4 && $3<=2.4 && $4 <=180 && $4>=-180) {print $0,3}}' tser_r1_r2_phi_si.dat > s3.dat
awk '{if($2 <=2.4 && $3>=2.4 && $4 <=180 && $4>=-180) {print $0,4}}' tser_r1_r2_phi_si.dat > s4.dat

awk '{if($2 <=2.5 && $3<=2.5 && $4 <=0 && $4>=-180) print $0,1; else if($2 <=2.5 && $3<=2.5 && $4 <=180 && $4>=0) print $0,2;else if($2 >=2.5 && $3<=2.5 && $4 <=180 && $4>=-180) print $0,4; else if($2 <=2.5 && $3>=2.5 && $4 <=180 && $4>=-180) print $0,3;}' tser_r1_r2_phi_si.dat > clust2.5_si.dat






#Aluminium
awk '{if($1 <=2.6 && $2<=2.6 && $3 <=0 && $3>=-180) print $0,0;else if($1 <=2.6 && $2<=2.6 && $3 <=180 && $3>=0) print $0,1;else if($1 >=2.6 && $2<=2.6 && $3 <=180 && $3>=-180) print $0,2;else if($1 <=2.6 && $2>=2.6 && $3 <=180 && $3>=-180) print $0,3;}' al_r1_r2_phi_m180_p180.dat> clust1_a1.dat

awk '{if($1 <=2.6 && $2<=2.6 && $3 <=0 && $3>=-180) print $0,1;else if($1 <=2.6 && $2<=2.6 && $3 <=180 && $3>=0) print $0,0;else if($1 >=2.6 && $2<=2.6 && $3 <=180 && $3>=-180) print $0,2;else if($1 <=2.6 && $2>=2.6 && $3 <=180 && $3>=-180) print $0,3;}' al_r1_r2_phi_m180_p180.dat > clust2.6_al.dat


awk '{if($1 <=2.6 && $2<=2.6 && $3 <=0 && $3>=-180) {print $0,1}}' tser_r1_r2_phi_si.dat > a1.dat
awk '{if($1 <=2.6 && $2<=2.6 && $3 <=180 && $3>=0) {print $0,2}}' tser_r1_r2_phi_si.dat > a2.dat
awk '{if($1 >=2.6 && $2<=2.6 && $3 <=180 && $3>=-180) {print $0,3}}' tser_r1_r2_phi_si.dat > a3.dat
awk '{if($1 <=2.6 && $2>=2.6 && $3 <=180 && $3>=-180) {print $0,4}}' tser_r1_r2_phi_si.dat > a4.dat


#MFPTS-matrix_plotting
library(corrplot)
data<-read.table("mfpts_matrix.dat")
data.matrix(data)
corrplot(cor, is.corr = FALSE, method = "circle",, col=colorRampPalette(c("blue","red"))(20))


