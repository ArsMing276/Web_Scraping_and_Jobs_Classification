########################### clean data ###############################
setwd("F:/First20")
f = list.files()
g = list.files(f, full.names = TRUE)
rawdata = list()
for(i in 1:length(g)){
  a = readLines(g[i],encoding = "UTF-8")
  a = a[-length(a)]
  index = which(grepl("[[:alpha:]]",a))
  a = a[index]
  b = gsub("<[/[:alpha:]]+>", "", a)
  b = gsub("[[:digit:]]","",b)
  b = tolower(b)
  d = strsplit(b[1], " ")[[1]]
  d = tolower(d[d!=""])
  if(any(grepl("job",d))) d = d[1:(which(grepl("job",d))-1)]
  
  if(length(b) == 1) rawdata[[i]] = d
  else{
    body = b[2]
    if(grepl('summary:',body)){
      c = strsplit(body, 'summary:')[[1]][2]
      b[2] = strsplit(c, ':')[[1]][1]
    }
    else if (grepl('qualifications:', body)){
      c = strsplit(body, 'qualifications')[[1]][2]
      b[2] = strsplit(c, ':')[[1]][1]
    } 
    else b[2] = ''
    
    e = strsplit(b[2], ' ')[[1]]
    rawdata[[i]] = c(d,e)
    rawdata[[i]] = gsub("[[:punct:]]","",rawdata[[i]])
    rawdata[[i]] = rawdata[[i]][rawdata[[i]]!='']
  }
}
names(rawdata) = g

########################### freq table ###########################
#use a sample
load("D:/rawdata.rda")
m = 5000
set.seed(2)
index = sample(length(rawdata), m)
train = rawdata[index]
trainname = names(train) #names of files
traintable = sapply(train, table)
allwords = lapply(traintable, names)
allword = Reduce(union, allwords) # get all the words
#generate the freq data frame
#give job class first
filename = sapply(trainname, strsplit, "/")
jobclass = sapply(filename, "[", 1)
jobclass = gsub(" Jobs", "", jobclass)
a = matrix(numeric(m*length(allword)), m)
rownames(a) = trainname
colnames(a) = allword
wordfreq = as.data.frame(a)
wordfreq = cbind(jobclass, wordfreq)
for(i in 1:m){
  wordfreq[i,names(traintable[[i]])] = traintable[[i]]
}

########################### KNN ###############################
#use knn method
wordscale = scale(wordfreq[,-1])#scale first
n = nrow(wordfreq)
#split in to train and test data 
train = wordscale[1:(n/2),]
test = wordscale[(n/2+1):n,]
#get the distance between train and test 
distance = as.matrix(dist(wordscale, method = "canberra"))[-(1:nrow(train)), (1:nrow(train))]
#use myknn function to find the nearst one
index = myknn(distance, 1)
#correct rate
correctrate = mean(wordfreq[(n/2+1):n,1]==wordfreq[1:(n/2),1][index])

#my knn function
myknn = function(distance, k){
  a = matrix( , nrow(distance), k)
  for(i in 1:nrow(distance)){
    a[i,] = order(distance[i,])[1:k]
  }
  return(a)
}

############### clustering and supervised model ####################
clustering varabiles to reduce number of variables
km.out=kmeans(t(as.matrix(wordfreq[,-1])),100,nstart=20)
a = matrix(numeric(nrow(wordfreq)*100), nrow = nrow(wordfreq))
for(i in 1:100){
  a[,i] = rowMeans(wordfreq[,-1][,which(km.out$cluster==i), drop = FALSE])
}
a = as.data.frame(a)
a$jobclass = wordfreq$jobclass
#split into train and test data
traincl = a[1:(n/2),]
testcl = a[(n/2+1):n,]
#svm method
library(e1071)
svm.fit = svm(jobclass~., data = traincl)
svmpred = predict(svm.fit, testcl, type = "class")
mean(svmpred==testcl$jobclass)

#knn method
library(class)
knn.fit = knn(traincl[,-ncol(traincl)], testcl[,-ncol(testcl)], traincl$jobclass)
mean(knn.fit==testcl$jobclass)



###################### pca machine learning methods ##################
##PCA dimension reduction
pc = prcomp(wordfreq[,-1])
eigenvalue = (pc$sdev)^2
cumprop = sum(eigenvalue[1:60])/sum(eigenvalue)
cumprop ###60 principle components are enough, explain 90% variance
scrs = pc$x
wordpca = matrix(nrow = 2000, ncol = 60)
wordpca[1:2000,1:60] = scrs[1:2000,1:60]
wordpca = as.data.frame(wordpca)
wordpca$Y = wordfreq$jobclass ##make a new data frame with jobclass as Y and 60 principle components

###Random Forest
library('randomForest')
ranFst=randomForest(Y~., data = wordpca[1:1000,],  ntree = 100)
error.ranFst = ranFst$err.rate ###prediction error rate given by OOB estimation
error.ranFst[100,1]
pred.rf = predict(ranFst,newdata=wordpca[1001:2000,-61])
mean(pred.rf == wordpca[1001:2000,61])

###Bagging
library('ipred')
bagging = bagging(Y~., data = wordpca[1:1000,], nbagg = 100,coob = TRUE)
err.bag = bagging$err
err.bag
pred.bag = predict(bagging, newdata = wordpca[1001:2000,-61])
mean(pred.bag == wordpca[1001:2000,61])

###SVM
library(e1071)
svm.fit = svm(Y~., data = wordpca[1:1000,])
svmpred = predict(svm.fit, wordpca[1001:2000, -61], type = "class")
mean(svmpred == wordpca$Y[1001:2000])
