library(tidyverse)
library(randomForest) 
library(rsample) # for splitting data into train and test
library(CatEncoders) # label encoder for categorical variables
library(caret)
library(nnet) # multinomial regression
library(e1071) # support vector machine
data <- read.csv("WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")
data.copy <- read.csv("WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv") # this data will be changed then compared to original for reference
set.seed(123)
unique(data$Sales.Channel)
unique(data$State)
unique(data$Coverage)
unique(data$Education)
unique(data$EmploymentStatus)
# 1 DATA PREP
summary(data) # checking for facotrs > 53 levels to fix error in random forest
str(data)
as.Date(data$Effective.To.Date, "%m/%d/%Y")
#enc = OneHotEncoder$new()
# PREPROCESSING
# Note: Random forest can handle catogircal data so will do that first then rerun after preprocessing

data.copy <- data.copy[,-1] # dropping ID column
str(data) # data still has an issue with data since > 53 categories
data.copy <-  data.copy[ , -which(names(data.copy) %in% c("Effective.To.Date","Response"))] # dropping months

#SPLITTING DATA INTO TRAIN AND TEST
split <- initial_split(data.copy,prop = 0.7)
train <- training(split)
ytrain <- train[,"Sales.Channel"]
test <- testing(split)
ytest <- test[,"Sales.Channel"]
test <-  test[ , -which(names(test) %in% c("Sales.Channel"))]

forest100 <- randomForest(Sales.Channel~.,ntree =100,data = train)
forest250 <- randomForest(Sales.Channel~.,ntree =250,data = train)
forest <- randomForest(Sales.Channel~.,data = train)
summary(forest)
forest$confusion
forest$err.rate
pred.forest <- predict(forest,newdata = test)
pred100 <- predict(forest100,newdata = test)
pred250 <- predict(forest250,newdata = test)
confuse.matrix <- table(pred.forest,ytest)
confuse100 <- table(pred100,ytest)
confuse250 <- table(pred250,ytest)
# missclassifaction rate is 1 - sum diagnol of table/total
misclass.rate <- 1-sum(diag(confuse.matrix))/sum(summary(ytest))
misclass100 <- 1-sum(diag(confuse100))/sum(summary(ytest))
misclass250 <- 1-sum(diag(confuse250))/sum(summary(ytest))
varImp(forest)

data.copy <- data[,-1]
data.copy <- data.copy[ , -which(names(data.copy) %in% c("Effective.To.Date"))]
#SPLITTING DATA INTO TRAIN AND TEST
split <- initial_split(data.copy,prop = 0.7)
res.train <- training(split)
ytrain <- train[,"Response"]
test <- testing(split)
res.ytest <- test[,"Response"]
res.test <-  test[ , -which(names(test) %in% c("Response"))]

forest100 <- randomForest(Response~.,ntree =100,data = train)
forest250 <- randomForest(Response~.,ntree =250,data = train)
forest <- randomForest(Response~.,data = train)
pred.forest <- predict(forest,newdata = res.test)
pred100 <- predict(forest100,newdata = res.test)
pred250 <- predict(forest250,newdata = res.test)
confuse.matrix <- table(pred.forest,ytest)
confuse100 <- table(pred100,ytest)
confuse250 <- table(pred250,ytest)
# missclassifaction rate is 1 - sum diagnol of table/total
misclass.rate <- 1-sum(diag(confuse.matrix))/sum(summary(ytest))
misclass100 <- 1-sum(diag(confuse100))/sum(summary(ytest))
misclass250 <- 1-sum(diag(confuse250))/sum(summary(ytest))

# change catgorical variables into numeric for regression and SVM
le = LabelEncoder.fit(data.copy$Gender) # starts from 1 alphabetically not 0, F = 1, M = 2
data.copy$Gender <- transform(le,data.copy$Gender) # changing gender into numeric values
le = LabelEncoder.fit(data.copy$State) # starts from 1 not 0
data.copy$State <- transform(le,data.copy$State) # changing State into numeric values
le = LabelEncoder.fit(data.copy$Response) # starts from 1 not 0
data.copy$Response <- transform(le,data.copy$Response) # changing Response into numeric values
le = LabelEncoder.fit(data.copy$Coverage) # starts from 1 not 0
data.copy$Coverage <- transform(le,data.copy$Coverage) # changing Coverage into numeric values
le = LabelEncoder.fit(data.copy$Education) # starts from 1 not 0
data.copy$Education <- transform(le,data.copy$Education) # changing Education into numeric values
le = LabelEncoder.fit(data.copy$EmploymentStatus) # starts from 1 not 0
data.copy$EmploymentStatus <- transform(le,data.copy$EmploymentStatus) # changing Emp status into numeric values
le = LabelEncoder.fit(data.copy$Marital.Status) # starts from 1 not 0
data.copy$Marital.Status <- transform(le,data.copy$Marital.Status) # changing martial status into numeric values
le = LabelEncoder.fit(data.copy$Location.Code) # starts from 1 not 0
data.copy$Location.Code <- transform(le,data.copy$Location.Code) # changing location code into numeric values
le = LabelEncoder.fit(data.copy$Policy.Type) # starts from 1 not 0
data.copy$Policy.Type <- transform(le,data.copy$Policy.Type) # changing location code into numeric values
le = LabelEncoder.fit(data.copy$Policy) # starts from 1 not 0
data.copy$Policy <- transform(le,data.copy$Policy) # changing location code into numeric values
le = LabelEncoder.fit(data.copy$Renew.Offer.Type) # starts from 1 not 0
data.copy$Renew.Offer.Type <- transform(le,data.copy$Renew.Offer.Type) # changing location code into numeric values
le = LabelEncoder.fit(data.copy$Vehicle.Class) # starts from 1 not 0
data.copy$Vehicle.Class <- transform(le,data.copy$Vehicle.Class) # changing vehicle class into numeric values
le = LabelEncoder.fit(data.copy$Vehicle.Size) # starts from 1 not 0
data.copy$Vehicle.Size <- transform(le,data.copy$Vehicle.Size) # changing vehicle size into numeric values
le = LabelEncoder.fit(data.copy$Sales.Channel) # starts from 1 not 0
data.copy$Sales.Channel <- transform(le,data.copy$Sales.Channel) # changing sales channel into numeric values

# MODELLING
# Logistic regression first for sales channel then for response
full.reg <- multinom(Sales.Channel~State+Customer.Lifetime.Value+Coverage+Education
                +EmploymentStatus+Gender+Income+Location.Code+Marital.Status+Monthly.Premium.Auto
                +Months.Since.Last.Claim+Months.Since.Policy.Inception+Number.of.Open.Complaints+Number.of.Policies
                +Policy.Type+Policy+Renew.Offer.Type+Total.Claim.Amount+Vehicle.Class+Vehicle.Size,data = train)
summary(full.reg)
pred.full <- predict(full.reg,test)
confuse.full <- table(pred.full,ytest)
acc.full <- sum(diag(confuse.full))/sum(summary(ytest))
varImp(full.reg) # get most important variables based on absolute value of t statistic
simple.reg <- multinom(Sales.Channel~State+Customer.Lifetime.Value+Education
                +EmploymentStatus+Income+Location.Code+Monthly.Premium.Auto+Number.of.Policies
                +Months.Since.Last.Claim+Months.Since.Policy.Inception+Number.of.Open.Complaints
                +Total.Claim.Amount+Vehicle.Class,data = train) # use variables < e-01
summary(simple.reg)
pred.simple <- predict(simple.reg,test)
confuse.simple <- table(pred.simple,ytest)
acc.simple <- sum(diag(confuse.simple))/sum(summary(ytest))
# trying to predict response
full.reg.res <- glm(Response~State+Sales.Channel+Customer.Lifetime.Value+Coverage+Education
                +EmploymentStatus+Gender+Income+Location.Code+Marital.Status+Monthly.Premium.Auto
                +Months.Since.Last.Claim+Months.Since.Policy.Inception+Number.of.Open.Complaints+Number.of.Policies
                +Policy.Type+Policy+Renew.Offer.Type+Total.Claim.Amount+Vehicle.Class+Vehicle.Size,data = train)
summary(full.reg.res)
pred.full.res <- predict(full.reg.res,res.test)
confuse.full.res <- table(pred.full.res,res.ytest)
acc.full.res <- sum(diag(confuse.full.res))/sum(summary(res.ytest))
varImp(full.reg.res) # get most important variables based on absolute value of t statistic
# rerun with variables greater than value of 1
simple.reg.res <- glm(Response~Sales.Channel+Marital.Status
                  +Monthly.Premium.Auto+Number.of.Policies
                  +Renew.Offer.Type+Total.Claim.Amount+Vehicle.Size,data = train)
summary(simple.reg.res)
pred.simple.res <- predict(simple.reg.res,res.test)
confuse.simple.res <- table(pred.simple.res,res.ytest)
acc.simple.res <- sum(diag(confuse.simple.res))/sum(summary(res.ytest))
# in the end Logistic regression is not good for predicting sales.channel and response
# SVM
svm.full <- svm(Sales.Channel ~., data = train, kernel = "radial")
svm.pred.full <- predict(svm.full,test)
svm.confuse <- table(svm.pred.full,ytest)
svm.confuse
svm.acc <- sum(diag(svm.confuse))/sum(summary(ytest))
# accuarcy of 0 

# Will redo all models using onylt the top 5 most important varibles based on graph
simple.data <- data.copy[ , which(names(data.copy) %in% c("Income","Monthly.Premium.Auto",
                                                          "Total.Claim.Amount",
                                                          "Months.Since.Policy.Inception",
                                                          "Customer.Lifetime.Value","Response"))]
split <- initial_split(simple.data,prop = 0.7)
res.train <- training(split)
ytrain <- res.train[,"Response"]
test <- testing(split)
res.ytest <- test[,"Response"]
res.test <-  test[ , -which(names(test) %in% c("Response"))] 

simple.forest <- randomForest(Response~.,ntree = 25,data = res.train)
simple.pred <- predict(simple.forest,res.test)
confusionMatrix(simple.pred$class,res.ytest)
summary(simple.pred) # displays predicted values
simple.confuse <- table(simple.pred,res.ytest)
acc1 <- sum(diag(simple.confuse))/sum(summary(res.ytest)) # accuracy of 99%

simple.data <- data.copy[ , which(names(data.copy) %in% c("Income","Monthly.Premium.Auto",
                                                          "Total.Claim.Amount",
                                                          "Months.Since.Policy.Inception",
                                                          "Customer.Lifetime.Value","Sales.Channel"))]
split <- initial_split(simple.data,prop = 0.7)
train <- training(split)
ytrain <- train[,"Sales.Channel"]
test <- testing(split)
ytest <- test[,"Sales.Channel"]
test <-  test[ , -which(names(test) %in% c("Sales.Channel"))] 

simple.sales <- randomForest(Sales.Channel~.,ntree = 100,data = train)
simple.sales.pred <- predict(simple.sales,test)
simple.confuse <- table(simple.sales.pred,ytest)
acc1.2 <- sum(diag(simple.confuse))/sum(summary(ytest)) # accuracy of 40% used 10,25,50,75,100 trees

# logistic regression for Response
simple.glm <- multinom(Response~Income+Total.Claim.Amount+Monthly.Premium.Auto
                  +Months.Since.Policy.Inception+Customer.Lifetime.Value,data = res.train)
simple.glm.pred <- predict(simple.glm,res.test)
simple.confuse <- table(simple.glm.pred,res.ytest)
acc2 <- sum(diag(simple.confuse))/sum(summary(res.ytest)) # accuracy of 85-86%
# logistic regression for Sales Channel
simple.glm.sales <- multinom(Sales.Channel~Income+Total.Claim.Amount+Monthly.Premium.Auto
                  +Months.Since.Policy.Inception+Customer.Lifetime.Value,data = train)
simple.glm.pred.sales <- predict(simple.glm.sales,test)
simple.confuse <- table(simple.glm.pred.sales,ytest)
acc2.2 <- sum(diag(simple.confuse))/sum(summary(ytest))

# SVM for Response
svm.res <- svm(Response~.,data = res.train, kernel = "polynomial",degree = 2)
svm.res.pred <- predict(svm.res,res.test)
svm.res.confuse <- table(svm.res.pred,res.ytest)
acc3 <- sum(diag(svm.res.confuse))/sum(summary(res.ytest)) # same as glm
# SVM for Sales Channel
svm.sales <- svm(Sales.Channel~.,data = train, kernel = "polynomial",degree = 4)
svm.sales.pred <- predict(svm.sales,test)
svm.sales.confuse <- table(svm.sales.pred,ytest)
acc3.2 <- sum(diag(svm.sales.confuse))/sum(summary(ytest)) # accuracy is 38 %
