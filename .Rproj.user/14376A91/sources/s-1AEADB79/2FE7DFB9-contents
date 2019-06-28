#Automating  Loan Approval process----

options(scipen = 999)

#Loading packages----
library(tidyverse)          # Data manipulation
library(caret)               #machine learning
library(mice)                #Data imputation
library(DataExplorer)      #Checking missing data values
library(tidyquant)         #for plotting
library(lares)             # for model evaluation


# Loading data set---- 
#train dara set
train <- read.csv("Data/train.csv",header = T,na.strings = c("NA",""))

#test data set
test <- read.csv("Data/test.csv",header = T,na.strings = c("NA",""))    

# Combining train and test data set to enable easy manipulation of data set
test$Loan_Status <- 'T'

dat <- rbind(train,test)

table(dat$Loan_Status)

str(dat)

summary(dat)

label <- dat$Loan_Status

attach(dat)

# Data cleaning----
#Spotting missing values----
# Percentage of missing data per column
#Visualizing Number of missing data per column
plot_missing(dat)

# Credit History has the highest number of missing values, at 8.14%
# Loan_Amount_Term 2.2%
# LoanAmount 3.58%
# Self_Employed 5.21%
# Dependents 2.44%

# Imputing the missing values using caret package----
# removing Loan_ID as its not predictive
dat$Loan_ID <- NULL

# Converting Credit_History into factor variable
dat$Credit_History <- as.factor(dat$Credit_History)

# Using mice to impute missing values
sum(is.na(dat))

# There are 233 missing values

set.seed(100)

dat_impute <- mice(dat, m = 5, maxit=10)

dat <- complete(dat_impute, 3)

# Checking that all missing values have been imputed
anyNA(dat)

# All missing values have been imputed

# Looking at the structure of the clean train data set----
str(dat)

# Coding credity history to yes and no
dat$Credit_History <- ifelse(dat$Credit_History == 1, 'Y', 'N')

dat$Credit_History <- as.factor(dat$Credit_History)

str(dat)

# Looking at summary of data set----
summary(dat[1:614,])

#Exploratory Data Analysis----
#1. Dependent variable, Loan Status----
table(dat[1:614,]$Loan_Status)

prop.table(table(dat[1:614,]$Loan_Status))

# Majority of loan applicants had their loan applications approved, at rate of 68%
# Its imbalanced and hence we use other evaluation metrics other than accuracy, e.g ROC and 
# sensitivity

# 2. Factor variables----
plot_bar(dat[1:614,])

# From the bar plots, majority of loan applicants were male, married, had no dependents,
# graduates, not self employed and Credit_History met guidelines

#3. Numeric variables----
plot_histogram(dat[1:614,])

# From the histograms, ApplicantIncome, CoapplicantIncome and LoanAmount are positively 
# skewed and majority of Loan_Amount_Term is 360 months

# Advanced Exploratory Data Analysis----
# Looking for correlations
cor(dat[1:614,6:9])

# LoanAmount and ApplicantIncome are positively correlated, as we expect applicants with high
# income to be able to secure high loan amounts

#1. Loan_Status, dependent variable----
# Looking at the distribution
table(dat[1:614,]$Loan_Status)

# Target variable is imbalanced with a ratio of about 1:2,this means we choose
# model evaluation methods critically,accurancy may be not the best, we will
# use other other methods like sensitivity, specificity and AUC

#2. Gender----
# Looking at the distribution
table(dat[1:614,]$Gender)

# Most of the applicants were male

# Testing for independence with Loan_Status
chisq.test(dat[1:614,]$Gender, dat[1:614,]$Loan_Status)

# A very large p-value > 5% confirms independence between genger and Loan_Status

# Visualizing
ggplot(dat[1:614,],aes(Gender,fill = Loan_Status)) +
  geom_bar(position = "fill") +
  ggtitle("Loan_Status By Gender") +
  xlab("\nGender") +
  ylab("Proportion") +
  theme_bw()

# Rate of loan approval across the gender is constant even though majority of
# loan applicants were males, this confirms gender was not determinant of loan approval

#3. Married----
#Looking at the distribution
table(dat[1:614,]$Married)

#Most of loan applicants were married.

# Testing for independence with Loan_Status
chisq.test(dat[1:614,]$Married, dat[1:614,]$Loan_Status)

# A small p-value of < 5% confirms dependence between marital status and Loan_Status

#Visualizing
ggplot(dat[1:614,],aes(Married,fill=Loan_Status))+
  geom_bar(position = 'fill')+
  ggtitle("Loan Approval Rate By Marital Status")+
  xlab("\nMarried")+
  ylab("Proportion") +
  theme_bw()

# Married loan applicants had a higher rate of loan approval compared to unmarried 
# applicants

#4. Dependents----
# Looking at the distributions

table(dat[1:614,]$Dependents)

# Most of loan applicants had zero dependents 
# Testing for independence with Loan_Status
chisq.test(dat[1:614,]$Dependents, dat[1:614,]$Loan_Status)

# A very large p-value > 5%  cofirms independence between Dependents and Loan_Status

#Visualizing
ggplot(dat[1:614,],aes(Dependents,fill=Loan_Status))+
  geom_bar(position = "fill")+
  ggtitle("Loan Approval Rate By Number of Dependents")+
  xlab("\nDependents")+
  ylab("Proportion")+
  theme_bw()

# Loan status seems constant across all levels of number of Dependents

#5. Education----
#Looking at the distribution
table(dat[1:614,]$Education)

#Most of loan applicants were graduates
# Testing for independence with Loan_Status
chisq.test(dat[1:614,]$Education, dat[1:614,]$Loan_Status)

# A p-value of 4% confirms dependence between Education and Loan_Status

# Visualizing
ggplot(dat[1:614,],aes(Education,fill=Loan_Status))+
  geom_bar(position = "fill")+
  ggtitle("Loan Approval Rate By Level of Education")+
  xlab("Education")+
  ylab("Proportion")+
  theme_bw()

#Graduates had a higher rate of loan approval compared to non graduates

#6. Self Employed----
# Looking at distribution
table(dat[1:614,]$Self_Employed)

#Most of loan applicant were not self employed.
# Testing for independence with Loan_Status
chisq.test(dat[1:614,]$Self_Employed, dat[1:614,]$Loan_Status)

# A very large p-value of 100% confirms independence

# Visualizing
ggplot(dat[1:614,],aes(Self_Employed,fill=Loan_Status))+
  geom_bar(position = "fill")+
  ggtitle("Loan Approval Rate By Self Employed")+
  xlab("\nSelf Employed")+
  ylab("Proportion")+
  theme_bw()

#Loan Approval rate is constants across all levels of self_employed even though 
# majority of loan applicants were not self employed.This confirms independence 
# between self employed and loan status

#7. ApplicantIncome----
#Looking at the summary
summary(dat[1:614,]$ApplicantIncome)

# ApplicantIncome is positively skewed
# Testing for equal means
t.test(ApplicantIncome~Loan_Status, data=dat[1:614,])

# A very large p-value of 92% confirms means are equal in two categories

#Visualizing by taking log10 of of ApplicantIncome
ggplot(dat[1:614,],aes(Loan_Status,log10(ApplicantIncome),fill=Loan_Status))+
  geom_boxplot()+
  ggtitle(" A boxplot of ApplicantIncome by Loan Approval")+
  xlab("\nLoan Status")+
  ylab("ApplicantIncome (log10 scale)")+
  theme_bw()

#The two box plots appears similar, suggesting ApplicantIncome is not 
# a determinant of loan approval 

# Looking at summary statistics across the two levels of loan status by applicantincome
dat[1:614,] %>% group_by(Loan_Status) %>% select(ApplicantIncome,Loan_Status)%>%
  summarise(average= mean(ApplicantIncome),Sd=sd(ApplicantIncome),
            Md = median(ApplicantIncome))

# Summary statistics from the two levels of loan status are almost similar suggesting
# ApplicantIncome is independent of loan status


#8. CoapplicantIncome----
#Looking at the summary
summary(dat[1:614,]$CoapplicantIncome)

# Testing for equal means
t.test(CoapplicantIncome~Loan_Status, data=dat[1:614,])

# A very large p-value of 26% confirms means are equal in two categories

#Visualizing
ggplot(dat[1:614,],aes(Loan_Status,log10(CoapplicantIncome+1),fill=Loan_Status))+
  geom_boxplot()+
  ggtitle(" A boxplot of CoApplicantIncome by Loan Approval")+
  xlab("\nLoan Status")+
  ylab("CoApplicantIncome (log10 scale)")+
  theme_bw()


dat[1:614,] %>% group_by(Loan_Status) %>% select(CoapplicantIncome,Loan_Status)%>%
  summarise(average= mean(CoapplicantIncome),Sd=sd(CoapplicantIncome),
            Md = median(CoapplicantIncome))

#Combining Applicantincome and coapplicantincome to see if the new variable is 
# predictive
# Testing correlation between ApplicantIncome and CoapplicantIncome
cor(dat[1:614,]$ApplicantIncome,dat[1:614,]$CoapplicantIncome)

# They have a weak and negative correlation

#TotalIncome----
dat$TotalIncome <- dat$ApplicantIncome + dat$CoapplicantIncome

#Looking at summary
summary(dat[1:614,]$TotalIncome)

# TotalIncome is positively skewed
# Testing for equal means
t.test(TotalIncome~Loan_Status, data=dat[1:614,])

# A large p-value of 49% confirms equal means between the two groups

# Visualizing by taking log10 of TotalIncome
ggplot(dat[1:614,],aes(Loan_Status,log10(TotalIncome),fill=Loan_Status))+
  geom_boxplot()+
  ggtitle(" A boxplot of TotalIncome by Loan Approval")+
  xlab("\nLoan Status")+
  ylab("TotalIncome (log10 scale)")+
  theme_bw()

# The two boxplots happear similar, suggesting new feature is not predictive

#Checking if having coapplication had loan request approved----
dat$Coapplicant <- ifelse(dat$CoapplicantIncome > 0, 'Yes', 'No')

table(dat[1:614,]$Coapplicant)

# Majority had Coapplicant income
# Testing for independence----
chisq.test(dat[1:614,]$Coapplicant, dat[1:614,]$Loan_Status)

# A p-value of 8% confirms independence

# Visualing
ggplot(dat[1:614,],aes(Coapplicant,fill=Loan_Status))+
  geom_bar(position = "fill")+
  ggtitle("Loan Approval Rate By Coapplicant")+
  xlab("\nCoapplicant ")+
  ylab("Proportion")+
  theme_bw()


#9. LoanAmount----
#Looking at summary
summary(dat[1:614,]$LoanAmount)

# Testing for equal means
t.test(LoanAmount~Loan_Status, data=dat[1:614,])

# A p-value of 54% confirms equal means

# Visualizing
ggplot(dat[1:614,],aes(Loan_Status,log10(LoanAmount),fill=Loan_Status))+
  geom_boxplot()+
  ggtitle(" A boxplot of LoanAmount by Loan status")+
  xlab("\nLoan Status")+
  ylab("Loan Amount")+
  theme_bw()

# The two boxplots happears the same, suggesting not predictive

# Looking at the summary across the levels
dat[1:614,] %>% group_by(Loan_Status) %>% select(LoanAmount,Loan_Status)%>%
  summarise(average= mean(LoanAmount),Sd=sd(LoanAmount), median = median(LoanAmount))

# summary statistics are similar across all levels of loan status

#10. Loan Amount Term----
#Looking at the distribution
summary(dat[1:614,]$Loan_Amount_Term)

# Testing for equal means
t.test(Loan_Amount_Term~Loan_Status, data=dat[1:614,])

# A very large p-value of 59% confirms equal means

# Visualizing 
ggplot(dat[1:614,],aes(Loan_Status,log10(Loan_Amount_Term),color=Loan_Status))+
  geom_boxplot()+
  ggtitle("Loan_Status By Loan Amount Term")+
  xlab("\nLoan Status")+
  ylab("Loan Amount Term in months(Log10 scale")+
  theme_bw()

# The two boxplot happear similar, suggesting not predictive


#11. Credit History----
# Looking at the distribution
table(dat[1:614,]$Credit_History)

# Most of the loan applicants had good credit history

# Visualizing
ch <- ggplot(dat[1:614,],aes(Credit_History,fill=Loan_Status))+
  geom_bar(position = "fill")+
  ggtitle("Loan Approval Rate By Credity History")+
  xlab("\nCredit History, Meeting Guidelines")+
  ylab("Loan Approval Rate")+
  theme_tq()+
  scale_fill_tq()+
  labs(subtitle="Loan Applicants whose credit history met guidelines had loan approval rate 
  above  75% and below 10% otherwise")+
  theme(legend.position = "top",
        plot.title = element_text(hjust=0.5, face='bold'),
        plot.subtitle=element_text(hjust=0.5))

ch

# From the plot, applicant with bad Credit_History had very little chance of
# getting their loan request approved
# Testing for independence
chisq.test(dat[1:614,]$Credit_History, dat[1:614,]$Loan_Status)

# A very small p-value confirms dependence between Credit_History and loan 
# approval

#12.Property Area----
# Looking at the distribution
table(dat[1:614,]$Property_Area)

# Most of loan applicants were from semiurban area  
# Testing for independence between property area and loan status
chisq.test(dat[1:614,]$Property_Area, dat[1:614,]$Loan_Status)

# A very small p-value of 0.2136% confirms dependence between property area and loan status

# Visualizing
ggplot(dat[1:614,],aes(Property_Area,fill=Loan_Status))+
  geom_bar(position = "fill")+
  ggtitle("Loan_Status By Property Area")+
  xlab("\nProperty Area")+
  ylab("Proportion")+
  theme_bw()

#From the plot, loan applicant from semiurban areas had high chance of getting
# their loan request approved.

#Advanced Exploratory data analysis----

# None of variables has near zero variance

#Loan Amount, Loan amount term, credity history, Property Area----
ggplot(dat[1:614,], aes(Loan_Amount_Term, LoanAmount, color=Loan_Status))+
  geom_point()+
  ggtitle("Scatter plot of Loan Amount and Loan Term by Credity History and Property Area")+
  xlab('\nLoan Term')+
  ylab("Loan Amount")+
  facet_grid(Credit_History ~ Property_Area)+
  theme_tq()+
  theme_bw()+
  theme(legend.position = "top")

# Credit_History seems to very predictive

#Applicant Income, loan amount, credity history, property area----
ggplot(dat[1:614,], aes(log10(ApplicantIncome),log10(LoanAmount), color=Loan_Status))+
  geom_point()+
  ggtitle("Scatter plot of ApplicantIncome and Loan Amount by Prpperty Area and Credity History")+
  xlab('\nApplicant Income')+
  ylab("Loan Amount")+
  facet_grid(Credit_History~Property_Area)+
  theme_tq()+
  theme_bw()+
  theme(legend.position = "top")

# Nothing striking from the plot

# ApplicantIncome and coapplicant income and property area and credit history----
ggplot(dat[1:614,], aes(ApplicantIncome, CoapplicantIncome, color=Loan_Status))+
  geom_point()+
  ggtitle("Scatter plot of Coapplicant income and Applicant income by property area and credit history")+
  xlab('\nApplicant Income')+
  ylab("Coapplicant Income")+
  facet_grid(Credit_History~Property_Area)+
  theme_tq()+
  theme_bw()+
  theme(legend.position = 'top')

# Credit_History seems to be only determining Loan_Status 

# Data preprocessing, removing skewness and ranging between 0 and 1----
str(dat)

dat$Coapplicant <- as.factor(dat$Coapplicant)

str(dat)

dat_pre <- preProcess(dat, method= c("YeoJohnson","range"))

dat <- predict(dat_pre, newdata=dat)

str(dat)

summary(dat)

#Creating Dummy Variables----
 credit <- dat$Credit_History 

dat$Credit_History <- NULL

str(dat)

dat_dummy <- dummyVars(Loan_Status~., data=dat)

dat <- predict(dat_dummy , newdata=dat)

dat <- as.data.frame(dat)

dat$Credity_History <- credit

str(dat)


summary(dat)

# Removing nearzero variance variables-----

nzv <- nearZeroVar(dat[1:614,],saveMetrics = TRUE)

nzv

# None of the variable has near zero variance

#Data Subsetting----
# Subsetting data to only include predictive variables
names(dat)


dat$Loan_Status <- as.factor(label)

name <- make.names(names=names(dat))

names(dat) <- name

str(dat)

summary(dat)

names(dat)

deselect <- c("Gender.Female","Gender.Male","Married.No","Married.Yes","Dependents.0",
              "Dependents.1","Dependents.2","Dependents.3.","Education.Graduate",
              "Education.Not.Graduate","Self_Employed.No","Self_Employed.Yes",
              "Property_Area.Urban","Property_Area.Rural","Coapplicant.Yes",
              "Coapplicant.No","Property_Area.Semiurban","Loan_Amount_Term")
              
dat_clean <- dat %>% select(-deselect)

new_train <- dat_clean[1:614,]
new_test <- dat_clean[615:981,]


#Data Splitting----
# We split the train data set into training and validating data sets
set.seed(100)
index <- createDataPartition(new_train$Loan_Status,times = 1,p=0.75,list = FALSE)
training <- new_train[index,]
validating <- new_train[-index,]

training$Loan_Status <- factor(training$Loan_Status)
validating$Loan_Status <- factor(validating$Loan_Status)

table(training$Loan_Status)

#Cross Validation----
# Cross validation helps our model perform better on unseen data set by reducing
# overfitting and underfitting
# Creating multifolds
set.seed(100)
k <- createMultiFolds(training$Loan_Status, k =10, times=10)

# Setting traincontrol
tr_control <- trainControl(method = "repeatedcv",number = 10,repeats = 10,index=k,
                           classProbs = T,savePredictions = TRUE,sampling='up', 
                           summaryFunction = twoClassSummary,verboseIter = FALSE)
                           

#Training a random forest model----

set.seed(100)

rf_model <- train(Loan_Status~.,data=training,method="rf",
                     trControl=tr_control,metric="ROC",verbose=FALSE)
                     
rf_model

rf_model$finalModel

varImp(rf_model)



#Testing the model-----
# Making predictions
pred <- predict(rf_model,newdata = validating)



# Creating a confusion matrix 
confusionMatrix(pred,validating$Loan_Status)


#ROC-AUC----
pred1 <- ifelse(pred=="N",0,1)
score1 <- ifelse(validating$Loan_Status=="N",0,1)

# plotting AUC
mplot_roc(tag=score1,score=pred1)

# We have an AUC of 78% for random forest model


# Making predictions on test data set for submission----


# Making predictions----
predict_rf <- predict(rf_model, newdata = new_test)

# Making a data frame for submission----
submit_rf <- data.frame(Loan_ID = test$Loan_ID,Loan_Status=predict_rf)

# Writing a csv file for submission
write.csv(submit_rf,file = "RF_6_19_28.csv",row.names = FALSE)


# We have a score of 79.861% at Analytics Vidhya



