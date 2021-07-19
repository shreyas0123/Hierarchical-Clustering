################################ problem1 #############################
#load the dataset
install.packages('readxl')
library(readxl)
my_data <- read_excel("C:\\Users\\DELL\\Downloads\\EastWestAirlines.xlsx",sheet = 2)
new_data <- my_data[ ,c(1:12)]
summary(new_data)

#remove duplicated data
dupl <- duplicated(new_data)
sum(dupl)
new_data <- new_data[!dupl ,]

#univariate analysis
#boxplot
#boxplot for some features of df norm_data
boxplot(new_data$Balance)
boxplot(new_data$Qual_miles)
boxplot(new_data$cc1_miles)
boxplot(new_data$cc2_miles)
boxplot(new_data$cc3_miles)
boxplot(new_data$Bonus_miles)
boxplot(new_data$Bonus_trans)
boxplot(new_data$Flight_miles_12mo)
boxplot(new_data$Flight_trans_12)
boxplot(new_data$Days_since_enroll)

#detection of outliers and use winzorisation technique
cunt_balance <- quantile(new_data$Balance , probs = c(.25 , .75))
winso_balance <- quantile(new_data$Balance , probs = c(.01 ,.90) , na.rm = TRUE)
H_balance <- 1.5*IQR(new_data$Balance ,na.rm = TRUE)
new_data$Balance[new_data$Balance<(cunt_balance[1]-H_balance)] <- winso_balance[1]
new_data$Balance[new_data$Balance>(cunt_balance[2]+H_balance)] <- winso_balance[2]
boxplot(new_data$Balance)

#detection of outliers and use winzorisation technique
cunt_bonus_miles <- quantile(new_data$Bonus_miles , probs = c(.25 , .75))
winso_bonus_miles <- quantile(new_data$Bonus_miles , probs = c(.01 ,.99) , na.rm = TRUE)
H_bonus_miles <- 1.5*IQR(new_data$Bonus_miles ,na.rm = TRUE)
new_data$Bonus_miles[new_data$Bonus_miles<(cunt_bonus_miles[1]-H_bonus_miles)] <- winso_bonus_miles[1]
new_data$Bonus_miles[new_data$Bonus_miles>(cunt_bonus_miles[2]+H_bonus_miles)] <- winso_bonus_miles[2]
boxplot(new_data$Bonus_miles)

#detection of outliers and use winzorisation technique
cunt_bonus_trans <- quantile(new_data$Bonus_trans , probs = c(.25 , .75))
winso_bonus_trans <- quantile(new_data$Bonus_trans , probs = c(.01 ,.99) , na.rm = TRUE)
H_bonus_trans <- 1.5*IQR(new_data$Bonus_trans ,na.rm = TRUE)
new_data$Bonus_trans[new_data$Bonus_trans<(cunt_bonus_trans[1]-H_bonus_trans)] <- winso_bonus_trans[1]
new_data$Bonus_trans[new_data$Bonus_trans>(cunt_bonus_trans[2]+H_bonus_trans)] <- winso_bonus_trans[2]
boxplot(new_data$Bonus_trans)

#detection of outliers and use winzorisation technique
cunt_flight_miles_12mo <- quantile(new_data$Flight_miles_12mo , probs = c(.25 , .75))
winso_flight_miles_12mo <- quantile(new_data$Flight_miles_12mo , probs = c(.01 ,.85) , na.rm = TRUE)
H_winso_flight_miles_12mo<- 1.5*IQR(new_data$Flight_miles_12mo ,na.rm = TRUE)
new_data$Flight_miles_12mo[new_data$Flight_miles_12mo<(cunt_flight_miles_12mo[1]-H_winso_flight_miles_12mo)] <- winso_flight_miles_12mo[1]
new_data$Flight_miles_12mo[new_data$Flight_miles_12mo>(cunt_flight_miles_12mo[2]+H_winso_flight_miles_12mo)] <- winso_flight_miles_12mo[2]
boxplot(new_data$Flight_miles_12mo)

#detection of outliers and use winzorisation technique
cunt_flight_trans_12<- quantile(new_data$Flight_trans_12 , probs = c(.25 , .75))
winso_flight_trans_12 <- quantile(new_data$Flight_trans_12 , probs = c(.01 ,.85) , na.rm = TRUE)
H_winso_flight_trans_12<- 1.5*IQR(new_data$Flight_trans_12 ,na.rm = TRUE)
new_data$Flight_trans_12[new_data$Flight_trans_12<(cunt_flight_trans_12[1]-H_winso_flight_trans_12)] <- winso_flight_trans_12[1]
new_data$Flight_trans_12[new_data$Flight_trans_12>(cunt_flight_trans_12[2]+H_winso_flight_trans_12)] <- winso_flight_trans_12[2]
boxplot(new_data$Flight_trans_12)


#histogram
#histogram for some features of df norm_data
hist(new_data$Balance)
hist(new_data$Qual_miles)
hist(new_data$cc1_miles)
hist(new_data$cc2_miles)
hist(new_data$cc3_miles)
hist(new_data$Bonus_miles)
hist(new_data$Bonus_trans)
hist(new_data$Flight_miles_12mo)
hist(new_data$Flight_trans_12)
hist(new_data$Days_since_enroll)

#dotchart
#dotchart for some features of df norm_data
dotchart(new_data$Balance)
dotchart(new_data$Qual_miles)
dotchart(new_data$cc1_miles)
dotchart(new_data$cc2_miles)
dotchart(new_data$cc3_miles)
dotchart(new_data$Bonus_miles)
dotchart(new_data$Bonus_trans)
dotchart(new_data$Flight_miles_12mo)
dotchart(new_data$Flight_trans_12)
dotchart(new_data$Days_since_enroll)

#bivariate analysis
#scatterplot for some features of df norm_data
install.packages("ggplot2")
library(ggplot2)
qplot(Balance,Qual_miles,data = new_data,color = cc1_miles,geom = "point")
qplot(cc2_miles,cc3_miles,data = new_data,color = Bonus_miles,geom = "point")
qplot(Flight_miles_12mo,Flight_trans_12,data = new_data,color = Days_since_enroll,geom = "point")

#normalise the data
norm_data <- scale(new_data[ ,c(1:10)])#excluding award
summary(norm_data)
sum(is.na(my_data))
sum(is.null(my_data))

#model building
Edist <- dist(norm_data,method = 'euclidean')
Hcl1 <- hclust(Edist,method = "complete")
Hcl2 <- hclust(Edist,method = "single")
Hcl3 <- hclust(Edist,method = "average")
Hcl4 <- hclust(Edist,method = "centroid")

#dendogram
plot(Hcl1, hang = -1)
plot(Hcl2, hang = -1)
plot(Hcl3, hang = -1)
plot(Hcl4, hang = -1)

clustered <- cutree(Hcl1,k=3)
rect.hclust(Hcl1, k = 3,border = "red")

Group <- as.matrix(clustered)
final <- data.frame(Group,new_data)

aggregate(new_data[ , 1:10],by = list(final$Group),FUN = mean)

install.packages("readr")
library(readr)
write_csv(final,"final.csv")
getwd()


################################ problem2 ######################################
#load the dataset
install.packages('readr')
library(readr)
my_data <- read_csv("C:\\Users\\DELL\\Downloads\\crime_data.csv")
new_data <- my_data[ ,c(2:5)]
summary(new_data)
sum(is.na(my_data))
sum(is.null(my_data))

#remove duplicated data
dupl <- duplicated(new_data)
sum(dupl)
new_data <- new_data[!dupl ,]

#univariate analysis
#boxplot
#boxplot for some features of df norm_data
boxplot(new_data$Murder)
boxplot(new_data$Assault)
boxplot(new_data$UrbanPop)
boxplot(new_data$"Customer Lifetime Value")

#detection of outliers and use winzorisation technique
cunt_Customer_Lifetime_Value<- quantile(new_data$"Customer Lifetime Value" , probs = c(.25 , .75))
winso_Customer_Lifetime_Value <- quantile(new_data$"Customer Lifetime Value" , probs = c(.01 ,.95) , na.rm = TRUE)
H_Customer_Lifetime_Value<- 1.5*IQR(new_data$"Customer Lifetime Value" ,na.rm = TRUE)
new_data$"Customer Lifetime Value"[new_data$"Customer Lifetime Value"<(cunt_Customer_Lifetime_Value[1]-H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[1]
new_data$"Customer Lifetime Value"[new_data$"Customer Lifetime Value">(cunt_Customer_Lifetime_Value[2]+H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[2]
boxplot(new_data$"Customer Lifetime Value")

#histogram
#histogram for some features of df norm_data
hist(new_data$Murder)
hist(new_data$Assault)
hist(new_data$UrbanPop)
hist(new_data$"Customer Lifetime Value")

#dotchart
#dotchart for some features of df norm_data
dotchart(new_data$Murder)
dotchart(new_data$Assault)
dotchart(new_data$UrbanPop)
dotchart(new_data$"Customer Lifetime Value")

#bivariate analysis
#scatterplot for some features of df norm_data
install.packages("ggplot2")
library(ggplot2)
qplot(Murder,Assault,data = new_data,color = UrbanPop,geom = "point")
qplot(Assault,Murder,data = new_data,color = new_data$"Customer Lifetime Value",geom = "point")

#normalise the data
norm_data <- scale(new_data[ ,2:4])
summary(norm_data)

#model building
Edist <- dist(norm_data,method = 'euclidean')
Hcl1 <- hclust(Edist,method = "complete")
Hcl2 <- hclust(Edist,method = "single")
Hcl3 <- hclust(Edist,method = "average")
Hcl4 <- hclust(Edist,method = "centroid")

#dendogram
plot(Hcl1, hang = -1)
plot(Hcl2, hang = -1)
plot(Hcl3, hang = -1)
plot(Hcl4, hang = -1)

clustered <- cutree(Hcl1,k=3)
rect.hclust(Hcl1, k = 3,border = "red")

Group <- as.matrix(clustered)
final <- data.frame(Group,new_data)

aggregate(new_data[2:5],by = list(final$Group),FUN = mean)

install.packages("readr")
library(readr)
write_csv(final,"final.csv")
getwd()

########################### problem3 ###################################
install.packages("readxl")
library(readxl)
my_data <- read_excel("C:\\Users\\DELL\\Downloads\\Telco_customer_churn.xlsx")
new_data <- my_data[ ,c(-2,-3)]
summary(new_data)
sum(is.na(my_data))
sum(is.null(my_data))
sum(columns(my_data))

#remove duplicated data
dupl <- duplicated(new_data)
sum(dupl)
new_data <- new_data[!dupl ,]

#univariate analysis
#boxplot
boxplot(new_data["Avg Monthly Long Distance Charges"])
boxplot(new_data["Avg Monthly GB Download"])
boxplot(new_data["Tenure in Months"])
boxplot(new_data["Avg Monthly Long Distance"])
boxplot(new_data["Monthly Charge"])
boxplot(new_data["Total Charges"])
boxplot(new_data["Total Refunds"])
boxplot(new_data["Total Extra Data Charges"])
boxplot(new_data["Total Long Distance Charges"])
boxplot(new_data["Total Revenue"])

#detection of outliers and use winzorisation technique
cunt_Avg_Monthly_GB_Download<- quantile(new_data$"Avg Monthly GB Download" , probs = c(.25 , .75))
winso_Avg_Monthly_GB_Download <- quantile(new_data$"Avg Monthly GB Download" , probs = c(.01 ,.90) , na.rm = TRUE)
H_Avg_Monthly_GB_Download<- 1.5*IQR(new_data$"Avg Monthly GB Download" ,na.rm = TRUE)
new_data$"Avg Monthly GB Download"[new_data$"Avg Monthly GB Download"<(cunt_Avg_Monthly_GB_Download[1]-H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[1]
new_data$"Avg Monthly GB Download"[new_data$"Avg Monthly GB Download">(cunt_Avg_Monthly_GB_Download[2]+H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[2]
boxplot(new_data$"Avg Monthly GB Download")


#detection of outliers and use winzorisation technique
cunt_Total_Long_Distance_Charges<- quantile(new_data$"Total Long Distance Charges" , probs = c(.25 , .75))
winso_Total_Long_Distance_Charges <- quantile(new_data$"Total Long Distance Charges" , probs = c(.01 ,.95) , na.rm = TRUE)
H_Total_Long_Distance_Charges<- 1.5*IQR(new_data$"Total Long Distance Charges" ,na.rm = TRUE)
new_data$"Total Long Distance Charges"[new_data$"Total Long Distance Charges"<(cunt_Total_Long_Distance_Charges[1]-H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[1]
new_data$"Total Long Distance Charges"[new_data$"Total Long Distance Charges">(cunt_Total_Long_Distance_Charges[2]+H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[2]
boxplot(new_data$"Total Long Distance Charges")

#detection of outliers and use winzorisation technique
cunt_Total_Revenue<- quantile(new_data$"Total Revenue" , probs = c(.25 , .75))
winso_Total_Revenue <- quantile(new_data$"Total Revenue" , probs = c(.01 ,.99) , na.rm = TRUE)
H_Total_Revenue<- 1.5*IQR(new_data$"Total Revenue" ,na.rm = TRUE)
new_data$"Total Revenue"[new_data$"Total Revenue"<(cunt_Total_Revenue[1]-H_Total_Revenue)] <- winso_Total_Revenue[1]
new_data$"Total Revenue"[new_data$"Total Revenue">(cunt_Total_Revenue[2]+H_Total_Revenue)] <- winso_Total_Revenue[2]
boxplot(new_data$"Total Revenue")

#bivariate analysis
#scatterplot for some features of df norm_data
install.packages("ggplot2")
library(ggplot2)
qplot(new_data$"Total Long Distance Charges",new_data$"Total Revenue",data = new_data,geom = "point")
qplot(new_data$"Avg Monthly GB Download",new_data$"Total Charges",data = new_data,color = new_data$"Total Refunds",geom = "point")

#creating dummies
install.packages("fastDummies")
library(fastDummies)

new_data_dummy <- dummy_cols(my_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

#normalise the data
norm_data <- scale(new_data_dummy)
summary(norm_data)

#model building
#use daisy() function
library(cluster)
my_data_dist <- daisy(norm_data , metric = "gower")
summary(my_data_dist)
my_data_dist <- as.matrix(my_data)

fit_my_data <- hclust(my_data_dist , method = "complete" )
plot(fit_my_data , hang = -1)
clust_my_data <- cuttree(fit_my_data , k = 3)
rect.hclust(fit_my_data , k = 3, border = "red")

Group <- as.matrix(clustered)
final <- data.frame(Group,new_data)

aggregate(new_data[1:30],by = list(final$Group),FUN = mean)

install.packages("readr")
library(readr)
write_csv(final,"final.csv")
getwd()

############################## problem4 ##############################
#load the dataset
install.packages('readr')
library(readr)
my_data <- read_csv("C:\\Users\\DELL\\Downloads\\AutoInsurance.csv")
new_data <- my_data[ ,c(-2)]
summary(new_data)
sum(is.na(my_data))
sum(is.null(my_data))

#remove duplicated data
dupl <- duplicated(new_data)
sum(dupl)
new_data <- new_data[!dupl ,]

#univariate analysis
#boxplot
#boxplot for some features of df norm_data
boxplot(new_data$"Customer Lifetime Value")
boxplot(new_data$"Income")
boxplot(new_data$"Monthly Premium Auto")
boxplot(new_data$"Months Since Last Claim")
boxplot(new_data$"Months Since Policy Inception")
boxplot(new_data$"Number of Open Complaints")
boxplot(new_data$"Number of Policies")
boxplot(new_data$"Total Claim Amount")

#detection of outliers and use winzorisation technique
cunt_Customer_Lifetime_Value<- quantile(new_data$"Customer Lifetime Value" , probs = c(.25 , .75))
winso_Customer_Lifetime_Value <- quantile(new_data$"Customer Lifetime Value" , probs = c(.01 ,.90) , na.rm = TRUE)
H_Customer_Lifetime_Value<- 1.5*IQR(new_data$"Customer Lifetime Value" ,na.rm = TRUE)
new_data$"Customer Lifetime Value"[new_data$"Customer Lifetime Value"<(cunt_Customer_Lifetime_Value[1]-H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[1]
new_data$"Customer Lifetime Value"[new_data$"Customer Lifetime Value">(cunt_Customer_Lifetime_Value[2]+H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[2]
boxplot(new_data$"Customer Lifetime Value")

#detection of outliers and use winzorisation technique
cunt_Monthly_Premium_Auto<- quantile(new_data$"Monthly Premium Auto" , probs = c(.25 , .75))
winso_Monthly_Premium_Auto <- quantile(new_data$"Monthly Premium Auto" , probs = c(.01 ,.95) , na.rm = TRUE)
H_Monthly_Premium_Auto<- 1.5*IQR(new_data$"Monthly Premium Auto" ,na.rm = TRUE)
new_data$"Monthly Premium Auto"[new_data$"Monthly Premium Auto"<(cunt_Monthly_Premium_Auto[1]-H_Monthly_Premium_Auto)] <- winso_Monthly_Premium_Auto[1]
new_data$"Monthly Premium Auto"[new_data$"Monthly Premium Auto">(cunt_Monthly_Premium_Auto[2]+H_Monthly_Premium_Auto)] <- winso_Monthly_Premium_Auto[2]
boxplot(new_data$"Monthly Premium Auto")

#detection of outliers and use winzorisation technique
cunt_Total_Claim_Amount<- quantile(new_data$"Total Claim Amount" , probs = c(.25 , .75))
winso_Total_Claim_Amount <- quantile(new_data$"Total Claim Amount" , probs = c(.01 ,.95) , na.rm = TRUE)
H_Total_Claim_Amount<- 1.5*IQR(new_data$"Total Claim Amount" ,na.rm = TRUE)
new_data$"Total Claim Amount"[new_data$"Total Claim Amount"<(cunt_Total_Claim_Amount[1]-H_Total_Claim_Amount)] <- winso_Total_Claim_Amount[1]
new_data$"Total Claim Amount"[new_data$"Total Claim Amount">(cunt_Total_Claim_Amount[2]+H_Total_Claim_Amount)] <- winso_Total_Claim_Amount[2]
boxplot(new_data$"Total Claim Amount")

#bivariate analysis
#scatterplot for some features of df norm_data
install.packages("ggplot2")
library(ggplot2)
qplot(new_data$"Total Claim Amount",new_data$"Monthly Premium Auto",data = new_data,geom = "point")
qplot(new_data$"Customer Lifetime Value",new_data$"Monthly Premium Auto",data = new_data,color = new_data$"Total Claim Amount",geom = "point")

#creating dummies
install.packages("fastDummies")
library(fastDummies)

new_data_dummies <- dummy_cols(new_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

#normalise the data
norm_data <- scale(new_data_dummies)
summary(norm_data)

#model building
Edist <- dist(norm_data,method = 'euclidean')
Hcl1 <- hclust(Edist,method = "complete")
Hcl2 <- hclust(Edist,method = "single")
Hcl3 <- hclust(Edist,method = "average")
Hcl4 <- hclust(Edist,method = "centroid")

#dendogram
plot(Hcl1, hang = -1)
plot(Hcl2, hang = -1)
plot(Hcl3, hang = -1)
plot(Hcl4, hang = -1)

clustered <- cutree(Hcl1,k=3)
rect.hclust(Hcl1, k = 3,border = "red")

Group <- as.matrix(clustered)
final <- data.frame(Group,new_data)

aggregate(new_data[1:24],by = list(final$Group),FUN = mean)

install.packages("readr")
library(readr)
write_csv(final,"final.csv")
getwd()


