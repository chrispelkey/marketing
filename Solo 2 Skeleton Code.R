## Chapter 1
load("stc-cbc-respondents-v3(1).RData")
ls()
str(resp.data.v3)
taskV3 <- read.csv("stc-dc-task-cbc -v3(1).csv", sep="\t")
str(taskV3)
require(dummies)
load("efCode.RData")
ls()
str(efcode.att.f)
str(efcode.attmat.f)
str(resp.data.v3)
str(taskV3)
apply(resp.data.v3[4:39], 2, function(x){tabulate(na.omit(x))}) 
task.mat <- as.matrix(taskV3[, c("screen", "RAM", "processor", "price", "brand")])
dim(task.mat)
head(task.mat)
X.mat=efcode.attmat.f(task.mat)  # Here is where we do effects coding
dim(X.mat)
head(X.mat)
pricevec=taskV3$price-mean(taskV3$price)
head(pricevec)
str(pricevec)
X.brands=X.mat[,9:11]
dim(X.brands)
str(X.brands)
X.BrandByPrice = X.brands*pricevec
dim(X.BrandByPrice)
str(X.BrandByPrice)
X.matrix=cbind(X.mat,X.BrandByPrice)
dim(X.matrix)
str(X.matrix)
X2.matrix=X.matrix[,1:2]
dim(X2.matrix)
det(t(X.matrix) %*% X.matrix)
ydata=resp.data.v3[,4:39]
names(ydata)
str(ydata)
ydata=na.omit(ydata)
str(ydata)
ydata=as.matrix(ydata)
dim(ydata)
zowner <- 1 * ( ! is.na(resp.data.v3$vList3) )
lgtdata = NULL
for (i in 1:424) { lgtdata[[i]]=list( y=ydata[i,],X=X.matrix )}
length(lgtdata)
str(lgtdata)

## Chapter 2
require(bayesm)
mcmctest=list(R=100000, keep=5)
Data1=list(p=3,lgtdata=lgtdata)
testrun1=rhierMnlDP(Data=Data1,Mcmc=mcmctest)
names(testrun1)
betadraw1=testrun1$betadraw
dim(betadraw1)
plot(1:length(betadraw1[350,10,]),betadraw1[350,10,])
plot(density(betadraw1[1,2,14001:20000],width=2))
summary(betadraw1[1,2,14001:20000])
betameansoverall <- apply(betadraw1[,,14001:20000],c(2),mean)
betameansoverall
perc <- apply(betadraw1[,,14001:20000],2,quantile,probs=c(0.05,0.10,0.25,0.5 ,0.75,0.90,0.95))
perc

## Chapter 3
zownertest=matrix(scale(zowner,scale=FALSE),ncol=1)
Data2=list(p=3,lgtdata=lgtdata,Z=zownertest)
testrun2=rhierMnlDP(Data=Data2,Mcmc=mcmctest)
dim(testrun2$deltadraw)
apply(testrun2$Deltadraw[14001:20000,],2,mean) 
apply(testrun2$Deltadraw[14001:20000,],2,quantile,probs=c(0.05,0.10,0.25,0.5 ,0.75,0.90,0.95))
betadraw2=testrun2$betadraw
dim(betadraw2)

## Chapter 4
betameans <- apply(betadraw1[,,14001:20000],c(1,2),mean)
str(betameans)
dim(betameans)
xbeta=X.matrix%*%t(betameans)
dim(xbeta)
xbeta2=matrix(xbeta,ncol=3,byrow=TRUE)
dim(xbeta2)
expxbeta2=exp(xbeta2)
rsumvec=rowSums(expxbeta2)
pchoicemat=expxbeta2/rsumvec
head(pchoicemat)
dim(pchoicemat)
custchoice <- max.col(pchoicemat)
str(custchoice)
head(custchoice)

ydatavec <- as.vector(t(ydata))
str(ydatavec)
table(custchoice,ydatavec)
require("pROC")
roctest <- roc(ydatavec, custchoice, plot=TRUE)
auc(roctest)
logliketest <- testrun1$loglike
mean(logliketest)

m <- matrix(custchoice, nrow =36,  byrow=F)
m2 <- t(m)
apply(m2, 2, function(x){tabulate(na.omit(x))})

##repeat this process for betadraw2##
betameans2 <- apply(betadraw2[,,14001:20000],c(1,2),mean)
str(betameans2)
dim(betameans2)
xbeta=X.matrix%*%t(betameans2)
dim(xbeta)
xbeta2=matrix(xbeta,ncol=3,byrow=TRUE)
dim(xbeta2)
expxbeta2=exp(xbeta2)
rsumvec=rowSums(expxbeta2)
pchoicemat=expxbeta2/rsumvec
head(pchoicemat)
dim(pchoicemat)
custchoice <- max.col(pchoicemat)
str(custchoice)
head(custchoice)

ydatavec <- as.vector(t(ydata))
str(ydatavec)
table(custchoice,ydatavec)
require("pROC")
roctest <- roc(ydatavec, custchoice, plot=TRUE)
auc(roctest)
logliketest <- testrun2$loglike
mean(logliketest)

m <- matrix(custchoice, nrow =36,  byrow=F)
m2 <- t(m)
apply(m2, 2, function(x){tabulate(na.omit(x))})

## Chapter 5
ex_scen <- read.csv("extra-scenarios(1).csv")
Xextra.matrix <- as.matrix(ex_scen[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9",
                                      "V10","V11","V12","V13","V14")])

betavec=matrix(betameansoverall,ncol=1,byrow=TRUE)
xextrabeta=Xextra.matrix%*%(betavec)
xbetaextra2=matrix(xextrabeta,ncol=3,byrow=TRUE)
dim(xbetaextra2)

expxbetaextra2=exp(xbetaextra2)
rsumvec=rowSums(expxbetaextra2)
pchoicemat=expxbetaextra2/rsumvec
pchoicemat

## We can predict the original 36 choice sets using the pooled model. 
## The code below, provides the probabilities as well as the frequencies for the 36 choice sets.

betavec=matrix(betameansoverall,ncol=1,byrow=TRUE)
xbeta=X.matrix%*%(betavec)
dim(xbeta)
xbeta2=matrix(xbeta,ncol=3,byrow=TRUE)
dim(xbeta2)
expxbeta2=exp(xbeta2)
rsumvec=rowSums(expxbeta2)
pchoicemat=expxbeta2/rsumvec
pchoicemat

pchoicemat2 <- round(pchoicemat*424,digits=0)
pchoicemat2
