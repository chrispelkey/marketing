setwd("~/Documents/PREDICT 450/Solo 3")
##################
load("XYZ_complete_customer_data_frame.RData")
ls()
mydata <- complete.customer.data.frame
str(mydata)
names(mydata)

table(mydata$RESPONSE16)

mydata$PRE2009SALES = mydata$LTD_SALES - mydata$YTD_SALES_2009
mydata$PRE2009TRANSACTIONS = mydata$LTD_TRANSACTIONS - 
  mydata$YTD_TRANSACTIONS_2009
mydata$cum15QTY <- mydata$QTY0 + mydata$QTY1 + mydata$QTY2 + mydata$QTY3 +
  mydata$QTY4 + mydata$QTY5 + mydata$QTY6 + mydata$QTY7 + mydata$QTY8 +
  mydata$QTY9 + mydata$QTY10 + mydata$QTY11 + mydata$QTY12 + mydata$QTY13 +
  mydata$QTY14 + mydata$QTY15

mydata$cum15TOTAMT <- mydata$TOTAMT0 + mydata$TOTAMT1 + mydata$TOTAMT2 + mydata$TOTAMT3 +
  mydata$TOTAMT4 + mydata$TOTAMT5 + mydata$TOTAMT6 + mydata$TOTAMT7 + mydata$TOTAMT8 +
  mydata$TOTAMT9 + mydata$TOTAMT10 + mydata$TOTAMT11 + mydata$TOTAMT12 + mydata$TOTAMT13 +
  mydata$TOTAMT14 + mydata$TOTAMT15

mydata$cum15RESPONSE <- mydata$RESPONSE0 + mydata$RESPONSE1 + mydata$RESPONSE2 + mydata$RESPONSE3 +
  mydata$RESPONSE4 + mydata$RESPONSE5 + mydata$RESPONSE6 + mydata$RESPONSE7 + mydata$RESPONSE8 +
  mydata$RESPONSE9 + mydata$RESPONSE10 + mydata$RESPONSE11 + mydata$RESPONSE12 + mydata$RESPONSE13 +
  mydata$RESPONSE14 + mydata$RESPONSE15

#mydata$salepercust <- mydata$PRE2009SALES/mydata$CENSUS_FACT1
mydata$salepertrans <- mydata$PRE2009SALES/mydata$PRE2009TRANSACTIONS
mydata$salepercamp <- mydata$cum15TOTAMT/mydata$TOTAL_MAIL_15

mean(mydata$salepertrans, trim = 0.01, na.rm = TRUE)
mean(mydata$salepercamp, trim = 0.05, na.rm = TRUE)

names(mydata)

require(rpart)
require(rpart.plot)
require(tree)
require(caTools)
require(ROCR)
require(ResourceSelection)

library(corrgram)
library(MASS)
library(randomForest)
library(inTrees)
library(pROC)
library(caret)
library(dplyr)

subdat <- subset(mydata, select=c("ZIP", "PRE2009SALES","PRE2009TRANSACTIONS", "cum15QTY", "MED_INC", 
                                 "cum15TOTAMT", "SUM_MAIL_16","TOTAL_MAIL_16","TOTAMT15","salepercamp",
                                 "ANY_MAIL_16","RESPONSE16","salepertrans", "HOMEOWNR", "NAT_INC",
                                 "M_HH_LEVEL", "LTD_SALES", "LTD_TRANSACTIONS", "BUYER_STATUS", "EXAGE"))

str(subdat)




subdat2 <- subset(subdat, ANY_MAIL_16 > 0)
str(subdat2)
#subdat2 <- subset(subdat, EXAGE > 0)
###subdat$ANY_MAIL_16 <- NULL 
##unitamtt <- mean(subdat2$TOTAMT)
subdat2$HOMEOWNR[subdat2$HOMEOWNR == ""] <- "N"
subdat2$HOMEOWNR[subdat2$HOMEOWNR == "U"] <- NA

subdat2$FRESPONSE16 <- factor(as.factor(subdat2$RESPONSE16))

head(subdat2)
subdat2$EXAGE[subdat2$EXAGE=="U"] <- NA
subdat3 <- na.omit(subdat2) 
head(subdat3)
str(subdat3)
subdat3$EXAGE <- as.numeric(subdat3$EXAGE)
mean(subdat3$EXAGE,na.rm=TRUE)

subdat3$ZIP <- factor(as.factor(subdat3$ZIP))
subdat3$M_HH_LEVEL <- factor(as.factor(subdat3$M_HH_LEVEL))
subdat3$BUYER_STATUS <- factor(as.factor(subdat3$BUYER_STATUS))

head(subdat3)
str(subdat3)

is.na(subdat3) <- sapply(subdat3, is.infinite)
subdat3 <- na.omit(subdat3)
str(subdat3)
table(subdat3$FRESPONSE16)

### note: 8562/9712 = 88% did not respond####

######## Logistic Regression Model #############

mylogit <- glm(RESPONSE16 ~ ZIP + PRE2009SALES + PRE2009TRANSACTIONS + cum15QTY + MED_INC +
               cum15TOTAMT + SUM_MAIL_16 + TOTAL_MAIL_16 + TOTAMT15 + salepercamp +
               ANY_MAIL_16 + salepertrans + NAT_INC + M_HH_LEVEL + LTD_SALES + 
               LTD_TRANSACTIONS + BUYER_STATUS + EXAGE, data = subdat3, family = "binomial")

summary(mylogit)
pvalue <- 1 - pchisq(425, df=8)
pvalue
###7065-6640=425; 9711-9703=8; deviance significant##
###Another GOF tets - Hosmer & lemeshow ###
hoslem.test(subdat3$RESPONSE16, fitted(mylogit))
pred2 <- predict(mylogit,data=subdat3, type="response")
head(pred2)
str(pred2)

pred2round <- round(pred2,0)
xtabs(~RESPONSE16 + pred2round, data = subdat3)
#### accuracy = 7931/8477 = 93.5%####


### ROCR curve
ROCRpred <- prediction(pred2,subdat3$RESPONSE16)
ROCRperf <- performance(ROCRpred, 'tpr', 'fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
abline(a=0,b=1)
auc(subdat3$RESPONSE16,pred2)

hist(pred2)
sort(pred2)

pred2df <- as.data.frame(pred2)
data_all <- cbind(subdat3,pred2)
str(data_all)
mean(data_all$salepertrans)
mean(data_all$salepertrans [data_all$pred2>0.6])
sum(data_all$pred2>0.6)

pred3 <- predict(mylogit, subdat, type="response")
pred3df <- as.data.frame(pred3)
data_all <- cbind(subdat,pred3)

###if the cutoff is 0.8 ###

########Random Forest model ###############

### putting '.' means includes all the variables
##rf1 <- randomForest(RESPONSE16 ~ .
##                    ,data=subdat3,importance=TRUE,ntree=100)
subdat3$HOMEOWNR <- factor(as.factor(subdat3$HOMEOWNR))
rf1 <- randomForest(FRESPONSE16 ~ ZIP + PRE2009SALES + PRE2009TRANSACTIONS + cum15QTY + MED_INC +
                      cum15TOTAMT + SUM_MAIL_16 + TOTAL_MAIL_16 + TOTAMT15 + salepercamp +
                      ANY_MAIL_16 + salepertrans +  NAT_INC + M_HH_LEVEL + LTD_SALES + HOMEOWNR +
                      LTD_TRANSACTIONS + BUYER_STATUS + EXAGE, data=subdat3, importance=TRUE, ntree=100)
summary(rf1)

##getTree(rf1,1,labelVar=TRUE)
##?getTree
print(rf1)
plot(rf1)
importance(rf1)
varImpPlot(rf1)
##how much each variable improves the prediction of its tree##
##compared to the exact same tree without that variable##


#get the prediction probabilities##
rf1p  <- predict(rf1, newdata=subdat3, type="response")
rf1p<-as.numeric(rf1p)
head(rf1p)
hist(rf1p)
sort(rf1p)
rf1.pred = prediction(rf1p, subdat3$RESPONSE16)
rf1.perf = performance(rf1.pred,"tpr","fpr")
plot(rf1.perf,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

auc(subdat3$RESPONSE16,rf1p)

#Make a confusion matrix and compute accuracy
##confusion.matrix <- table(rf1p, subdat3$RESPONSE16)
##confusion.matrix

rf1pround <- round(rf1p,0)
xtabs(~RESPONSE16 + rf1pround, data = subdat3)

## 7417+333/8477 = .914

rf1pdf <- as.data.frame(rf1p)
data_alldf <- cbind(subdat3,rf1pdf)
str(data_alldf)
head(data_alldf)
mean(data_alldf$salepertrans)
mean(data_alldf$salepertrans [data_alldf$rf1p>1.2])
sum(data_alldf$rf1p>1.2)
##############################




 

########Second Random Forest model ###############

### putting '.' means includes all the variables
##rf1 <- randomForest(RESPONSE16 ~ .
##                    ,data=subdat3,importance=TRUE,ntree=100)
rf2 <- randomForest(FRESPONSE16 ~ M_HH_LEVEL + PRE2009SALES + PRE2009TRANSACTIONS + cum15QTY + MED_INC +
                      cum15TOTAMT + TOTAL_MAIL_16 + TOTAMT15 + salepercamp +
                      ANY_MAIL_16 + salepertrans +  NAT_INC + LTD_SALES + 
                      LTD_TRANSACTIONS + BUYER_STATUS + EXAGE, data=subdat3, importance=TRUE, ntree=100)
summary(rf2)

##getTree(rf1,1,labelVar=TRUE)
##?getTree
print(rf2)
plot(rf2)
importance(rf2)
varImpPlot(rf2)
##how much each variable improves the prediction of its tree##
##compared to the exact same tree without that variable##


#get the prediction probabilities##
rf2p  <- predict(rf2, newdata=subdat3, type="response")
rf2p<-as.numeric(rf2p)
head(rf2p)
hist(rf2p)
sort(rf2p)
rf2.pred = prediction(rf2p, subdat3$RESPONSE16)
rf2.perf = performance(rf2.pred,"tpr","fpr")
plot(rf2.perf,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

auc(subdat3$RESPONSE16,rf2p)

#Make a confusion matrix and compute accuracy
##confusion.matrix <- table(rf1p, subdat3$RESPONSE16)
##confusion.matrix

rf2pround <- round(rf2p,0)
xtabs(~RESPONSE16 + rf2pround, data = subdat3)

## 7417+561/8477 = .941

rf2pdf <- as.data.frame(rf2p)
data_alldf2 <- cbind(subdat3,rf2pdf)
str(data_alldf2)
head(data_alldf2)
mean(data_alldf2$salepertrans)
mean(data_alldf2$salepertrans [data_alldf2$rf2p>1.2])
sum(data_alldf2$rf2p>1.2)
##############################


library(VIM)
aggr_plot <- aggr(subdat2, col=c('navyblue','red'), numbers = TRUE, labels =names(subdat2), cex.axis = .7, gap = 3, ylab = c("Histogram of missing data", "Pattern"))
aggr_plot
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(subdat2,2,pmiss)

