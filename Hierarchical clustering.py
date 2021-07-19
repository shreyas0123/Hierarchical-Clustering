################################### problem1 #######################################
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
airlines = pd.read_excel("C://Users//DELL//Downloads//EastWestAirlines.xlsx", sheet_name= "data" )
airlines.describe()
airlines.info()
airlines.isna().sum()
airlines.isnull().sum()

#to check duplicates
dups = airlines.duplicated()
sum(dups)

airl_drop = airlines.drop_duplicates()

#univariate analysis
#boxplot for some features of df airlines
plt.boxplot(airlines.Balance)
plt.boxplot(airlines.Qual_miles)
plt.boxplot(airlines.cc1_miles)
plt.boxplot(airlines.cc2_miles)
plt.boxplot(airlines.cc3_miles)
plt.boxplot(airlines.Bonus_miles)
plt.boxplot(airlines.Bonus_trans)
plt.boxplot(airlines.Flight_miles_12mo)
plt.boxplot(airlines.Flight_trans_12)
plt.boxplot(airlines.Days_since_enroll)

#histogram for some features of df airlines
plt.hist(airlines.Balance)
plt.hist(airlines.Qual_miles)
plt.hist(airlines.cc1_miles)
plt.hist(airlines.cc3_miles)

#detection of outliers(find the RM based on IQR)
IQR = airlines["Balance"].quantile(0.75) - airlines["Balance"].quantile(0.25)
lower_limit_balance = airlines["Balance"].quantile(0.25) - (IQR * 1.5)
upper_limit_balance = airlines["Balance"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["airlines_replaced"] = pd.DataFrame(np.where(airlines["Balance"] > upper_limit_balance, upper_limit_balance,
                                                      np.where(airlines["Balance"] < lower_limit_balance,lower_limit_balance,airlines["Balance"])))
sns.boxplot(airlines["airlines_replaced"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Bonus_miles"].quantile(0.75) - airlines["Bonus_miles"].quantile(0.25)
lower_limit_bonus_miles = airlines["Bonus_miles"].quantile(0.25) - (IQR * 1.5)
upper_limit_bonus_miles = airlines["Bonus_miles"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["airlines_replaced"] = pd.DataFrame(np.where(airlines["Bonus_miles"] > upper_limit_bonus_miles, upper_limit_bonus_miles,
                                                      np.where(airlines["Bonus_miles"] < lower_limit_bonus_miles,lower_limit_bonus_miles,airlines["Bonus_miles"])))
sns.boxplot(airlines["airlines_replaced"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Bonus_trans"].quantile(0.75) - airlines["Bonus_trans"].quantile(0.25)
lower_limit_Bonus_trans = airlines["Bonus_trans"].quantile(0.25) - (IQR * 1.5)
upper_limit_Bonus_trans = airlines["Bonus_trans"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["airlines_replaced"] = pd.DataFrame(np.where(airlines["Bonus_trans"] > upper_limit_Bonus_trans, upper_limit_Bonus_trans,
                                                      np.where(airlines["Bonus_trans"] < lower_limit_Bonus_trans,lower_limit_Bonus_trans,airlines["Bonus_trans"])))
sns.boxplot(airlines["airlines_replaced"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Flight_miles_12mo"].quantile(0.75) - airlines["Flight_miles_12mo"].quantile(0.25)
lower_limit_Flight_miles_12mo = airlines["Flight_miles_12mo"].quantile(0.25) - (IQR * 1.5)
upper_limit_Flight_miles_12mo = airlines["Flight_miles_12mo"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["airlines_replaced"] = pd.DataFrame(np.where(airlines["Flight_miles_12mo"] > upper_limit_Flight_miles_12mo, upper_limit_Flight_miles_12mo,
                                                      np.where(airlines["Flight_miles_12mo"] < lower_limit_Flight_miles_12mo,lower_limit_Flight_miles_12mo,airlines["Flight_miles_12mo"])))
sns.boxplot(airlines["airlines_replaced"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Flight_trans_12"].quantile(0.75) - airlines["Flight_trans_12"].quantile(0.25)
lower_limit_Flight_trans_12 = airlines["Flight_trans_12"].quantile(0.25) - (IQR * 1.5)
upper_limit_Flight_trans_12 = airlines["Flight_trans_12"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["Flight_trans_12"] = pd.DataFrame(np.where(airlines["Flight_trans_12"] > upper_limit_Flight_trans_12, upper_limit_Flight_trans_12,
                                                      np.where(airlines["Flight_trans_12"] < lower_limit_Flight_trans_12,lower_limit_Flight_trans_12,airlines["Flight_trans_12"])))
sns.boxplot(airlines["Flight_trans_12"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Days_since_enroll"].quantile(0.75) - airlines["Days_since_enroll"].quantile(0.25)
lower_limit_Days_since_enroll = airlines["Days_since_enroll"].quantile(0.25) - (IQR * 1.5)
upper_limit_Days_since_enroll = airlines["Days_since_enroll"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["Days_since_enroll"] = pd.DataFrame(np.where(airlines["Days_since_enroll"] > upper_limit_Days_since_enroll, upper_limit_Days_since_enroll,
                                                      np.where(airlines["Days_since_enroll"] < lower_limit_Days_since_enroll,lower_limit_Days_since_enroll,airlines["Days_since_enroll"])))
sns.boxplot(airlines["Days_since_enroll"]);plt.title("Boxplot");plt.show()

#bivariate analysis
#scatterplot

sns.scatterplot(airlines["Balance"],airlines["Qual_miles"])
sns.scatterplot(airlines["cc1_miles"],airlines["cc2_miles"])
sns.scatterplot(airlines["cc3_miles"],airlines["Bonus_miles"])
sns.scatterplot(airlines["Bonus_trans"],airlines["Bonus_trans"])

#normalization function
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
    
#normalising data frame (conidering the numerical part of the data)
df_norm = norm_fun(airlines.iloc[:, 0:])
df_norm.describe()

airlines.columns

#model building
#for creating dendogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

airlines_linkage = linkage(df_norm,method = "complete", metric = "euclidean")

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(airlines_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

airlines_linkage = linkage(df_norm,method = "single", metric = "euclidean")

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(airlines_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

airlines_linkage = linkage(df_norm,method = "average", metric = "euclidean")

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(airlines_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

airlines_linkage = linkage(df_norm,method = "centroid", metric = "euclidean")

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(airlines_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

#for creating dendogram
#now applying AgglomerativeClustering choosing 3 as clusters from the above dendogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np

airlines_complete = AgglomerativeClustering(n_clusters = 3,linkage = "complete",affinity = "euclidean").fit(df_norm)
airlines_complete.labels_
cluster_labels = pd.Series(airlines_complete.labels_)
airlines['clust'] = cluster_labels #creating a new coloumn and assingning cluster_labels values to it

airlines_single = AgglomerativeClustering(n_clusters = 3,linkage = "single",affinity = "euclidean").fit(df_norm)
airlines_single.labels_
cluster_labels = pd.Series(airlines_single.labels_)
airlines['clust'] = cluster_labels #creating a new coloumn and assingning cluster_labels values to it

airlines_average = AgglomerativeClustering(n_clusters = 3,linkage = "average",affinity = "euclidean").fit(df_norm)
airlines_average.labels_
cluster_labels = pd.Series(airlines_average.labels_)
airlines['clust'] = cluster_labels #creating a new coloumn and assingning cluster_labels values to it

airlines_centroid = AgglomerativeClustering(n_clusters = 3,linkage = "centroid",affinity = "euclidean").fit(df_norm)
airlines_centroid.labels_
cluster_labels = pd.Series(airlines_centroid.labels_)
airlines['clust'] = cluster_labels #creating a new coloumn and assingning cluster_labels values to it



airlines = airlines.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]]
airlines.head()

#Aggregate the mean of each cluster
airlines.iloc[: ,0:].groupby(airlines.clust).mean() 

#creating a csv file
airlines.to_csv("new_airlines.csv", encoding = "utf-8")
import os
os.getcwd()

####################################### problem2 ###############################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import numpy as np
import os
import seaborn as sns

crime_data = pd.read_csv("C:\\Users\\DELL\\Downloads\\crime_data.csv")
crime_data.describe()
crime_data.info()
crime_data.isna().sum()
crime_data.isnull().sum()
crime_data.columns

#to check duplicates
dups = crime_data.duplicated()
sum(dups)

crime_data = crime_data.drop_duplicates()
#univariate analysis
#boxplot
plt.boxplot(crime_data.Murder)
plt.boxplot(crime_data.Assault)
plt.boxplot(crime_data.UrbanPop)
plt.boxplot(crime_data.Rape)

#bivariate analysis
#scatterplot
sns.scatterplot(crime_data["Assault"],crime_data["Murder"])
sns.scatterplot(crime_data["UrbanPop"],crime_data["Rape"])

#detection of outliers(find the RM based on IQR)
IQR = crime_data["Rape"].quantile(0.75) - crime_data["Rape"].quantile(0.25)
lower_limit_Rape = crime_data["Rape"].quantile(0.25) - (IQR * 1.5)
upper_limit_Rape = crime_data["Rape"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
crime_data["Rape"] = pd.DataFrame(np.where(crime_data["Rape"] > upper_limit_Rape, upper_limit_Rape,
                                                      np.where(crime_data["Rape"] < lower_limit_Rape,lower_limit_Rape,crime_data["Rape"])))
sns.boxplot(crime_data["Rape"]);plt.title("Boxplot");plt.show()

#normalisation function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
#normalised data frame considering numerical part of the data
crime_data_norm = norm_func(crime_data.iloc[:,1:])
crime_data_norm.describe()

#model building
#dendogram
crime_data_linkage = linkage(crime_data_norm,method = "complete",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using complete linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(crime_data_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

crime_data_linkage = linkage(crime_data_norm,method = "single",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(crime_data_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


crime_data_linkage = linkage(crime_data_norm,method = "average",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using average linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(crime_data_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

crime_data_linkage = linkage(crime_data_norm,method = "centroid",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using centroid linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(crime_data_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

#now applying AgglomerativeClustering choosing 3 as clusters from the above dendogram
crime_data_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete',affinity ="euclidean").fit(crime_data_norm)
crime_data_complete.labels_
cluster_labels = pd.Series(crime_data_complete.labels_)
crime_data['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

crime_data_single = AgglomerativeClustering(n_clusters = 3, linkage = 'single',affinity ="euclidean").fit(crime_data_norm)
crime_data_single.labels_
cluster_labels = pd.Series(crime_data_single.labels_)
crime_data['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

crime_data_average = AgglomerativeClustering(n_clusters = 3, linkage = 'average',affinity ="euclidean").fit(crime_data_norm)
crime_data_average.labels_
cluster_labels = pd.Series(crime_data_average.labels_)
crime_data['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

crime_data_centroid = AgglomerativeClustering(n_clusters = 3, linkage = 'centroid',affinity ="euclidean").fit(crime_data_norm)
crime_data_centroid.labels_
cluster_labels = pd.Series(crime_data_centroid.labels_)
crime_data['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

crime_data = crime_data.iloc[:,[5,0,1,2,3,4]]
crime_data.head()

#Aggregate the mean of each cluster
crime_data.iloc[:,1:].groupby(crime_data.clust).mean()

#creating a csv file
crime_data.to_csv("new_crime_data.csv", encoding = "utf-8")
os.getcwd()

######################################### problem3 ###############################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import numpy as np
import seaborn as sns
import os

tel_com = pd.read_excel("C:\\Users\\DELL\\Downloads\\Telco_customer_churn.xlsx")
tel_com.describe()
tel_com.info()
tel_com.isna().sum()
tel_com.columns

#to check duplicates
dups = tel_com.duplicated()
sum(dups)

tel_com = tel_com.drop_duplicates()

#drop count,quarter coloumns
tel_com.drop(['Count','Quarter'],axis = 1,inplace = True)
tel_com.dtypes

#creat dummy variables on categorical data
tel_com_new = pd.get_dummies(tel_com)

#creating instance of one hot encoding
tel_com = pd.read_excel("C:\\Users\\DELL\\Downloads\\Telco_customer_churn.xlsx")
tel_com.describe()
tel_com.drop(['Count','Quarter'],axis = 1,inplace = True)
tel_com.dtypes

from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
one_hot_tel_com = pd.DataFrame(one_hot.fit_transform(tel_com).toarray())

#creating instance of labelencoder
tel_com = pd.read_excel("C:\\Users\\DELL\\Downloads\\Telco_customer_churn.xlsx")
tel_com.describe()
tel_com.drop(['Count','Quarter'],axis = 1,inplace = True)
tel_com.dtypes

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x = tel_com.iloc[: , :]

x['Customer ID'] = label_encoder.fit_transform(x['Customer ID'])
x['Count'] = label_encoder.fit_transform(x['Count'])
x['Quarter'] = label_encoder.fit_transform(x['Quarter'])
x['Referred a Friend'] = label_encoder.fit_transform(x['Referred a Friend'])
x['Number of Referrals'] = label_encoder.fit_transform(x['Number of Referrals'])
x['Tenure in Months'] = label_encoder.fit_transform(x['Tenure in Months'])
x['Offer'] = label_encoder.fit_transform(x['Offer'])
x['Phone Service'] = label_encoder.fit_transform(x['Phone Service'])
x['Avg Monthly Long Distance Charges'] = label_encoder.fit_transform(x['Avg Monthly Long Distance Charges'])
x['Multiple Lines'] = label_encoder.fit_transform(x['Multiple Lines'])
x['Internet Service'] = label_encoder.fit_transform(x['Internet Service'])
x['Internet Type'] = label_encoder.fit_transform(x['Internet Type'])
x['Avg Monthly GB Download'] = label_encoder.fit_transform(x['Avg Monthly GB Download'])
x['Online Security'] = label_encoder.fit_transform(x['Online Security'])
x['Online Backup'] = label_encoder.fit_transform(x['Online Backup'])
x['Device Protection Plan'] = label_encoder.fit_transform(x['Device Protection Plan'])
x['Premium Tech Support'] = label_encoder.fit_transform(x['Premium Tech Support'])
x['Streaming TV'] = label_encoder.fit_transform(x['Streaming TV'])
x['Streaming Movies'] = label_encoder.fit_transform(x['Streaming Movies'])
x['Streaming Music'] = label_encoder.fit_transform(x['Streaming Music'])
x['Unlimited Data'] = label_encoder.fit_transform(x['Unlimited Data'])
x['Contract'] = label_encoder.fit_transform(x['Contract'])
x['Paperless Billing'] = label_encoder.fit_transform(x['Paperless Billing'])
x['Payment Method'] = label_encoder.fit_transform(x['Payment Method'])
x['Monthly Charge'] = label_encoder.fit_transform(x['Monthly Charge'])
x['Total Charges'] = label_encoder.fit_transform(x['Total Charges'])
x['Total Refunds'] = label_encoder.fit_transform(x['Total Refunds'])
x['Total Refunds'] = label_encoder.fit_transform(x['Total Refunds'])
x['Total Long Distance Charges'] = label_encoder.fit_transform(x['Total Long Distance Charges'])
x['Total Revenue'] = label_encoder.fit_transform(x['Total Revenue'])

#univariate analysis
#boxplot
plt.boxplot(tel_com["Avg Monthly Long Distance Charges"])
plt.boxplot(tel_com["Count"])
plt.boxplot(tel_com["Number of Referrals"]) 
plt.boxplot(tel_com["Tenure in Months"])
plt.boxplot(tel_com["Avg Monthly Long Distance"])
plt.boxplot(tel_com["Avg Monthly GB Download"])
plt.boxplot(tel_com["Monthly Charge"])
plt.boxplot(tel_com["Total Charges"])
plt.boxplot(tel_com["Total Refunds"])
plt.boxplot(tel_com["Total Extra Data Charges"])
plt.boxplot(tel_com["Total Long Distance Charges"])
plt.boxplot(tel_com["Total Revenue"])

#detection of outliers(find the RM based on IQR)
IQR = tel_com["Number of Referrals"].quantile(0.75) - tel_com["Number of Referrals"].quantile(0.25)
lower_limit_Number_of_Referrals = tel_com["Number of Referrals"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = tel_com["Number of Referrals"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
tel_com["Number of Referrals"] = pd.DataFrame(np.where(tel_com["Number of Referrals"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(tel_com["Number of Referrals"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,tel_com["Number of Referrals"])))
sns.boxplot(tel_com["Number of Referrals"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = tel_com["Monthly Charge"].quantile(0.75) - tel_com["Monthly Charge"].quantile(0.25)
lower_limit_Number_of_Referrals = tel_com["Monthly Charge"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = tel_com["Monthly Charge"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
tel_com["Monthly Charge"] = pd.DataFrame(np.where(tel_com["Monthly Charge"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(tel_com["Monthly Charge"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,tel_com["Monthly Charge"])))
sns.boxplot(tel_com["Monthly Charge"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = tel_com["Total Revenue"].quantile(0.75) - tel_com["Total Revenue"].quantile(0.25)
lower_limit_Number_of_Referrals = tel_com["Total Revenue"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = tel_com["Total Revenue"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
tel_com["Total Revenue"] = pd.DataFrame(np.where(tel_com["Total Revenue"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(tel_com["Total Revenue"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,tel_com["Total Revenue"])))
sns.boxplot(tel_com["Total Revenue"]);plt.title("Boxplot");plt.show()

#bivariate analysis
#scatterplot
sns.scatterplot(tel_com["Number of Referrals"],tel_com["Monthly Charge"])
sns.scatterplot(tel_com["Number of Referrals"],tel_com["Total Revenue"])

#normalisation function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

tel_com_norm = norm_func(tel_com_new)

#model building
#dendogram
tel_com_new_linkage = linkage(tel_com_norm,method = "complete",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using complete linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(tel_com_new_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

tel_com_new_linkage  = linkage(tel_com_norm,method = "single",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(tel_com_new_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


tel_com_new_linkage = linkage(tel_com_norm,method = "average",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using average linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(tel_com_new_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

tel_com_new_linkage = linkage(tel_com_norm,method = "centroid",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using centroid linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(tel_com_new_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

#now applying AgglomerativeClustering choosing 3 as clusters from the above dendogram
tel_com_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete',affinity ="euclidean").fit(tel_com_norm)
tel_com_complete.labels_
cluster_labels = pd.Series(tel_com_complete.labels_)
tel_com['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

tel_com_single = AgglomerativeClustering(n_clusters = 3, linkage = 'single',affinity ="euclidean").fit(tel_com_norm)
tel_com_single.labels_
cluster_labels = pd.Series(tel_com_single.labels_)
tel_com['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

tel_com_average = AgglomerativeClustering(n_clusters = 3, linkage = 'average',affinity ="euclidean").fit(tel_com_norm)
tel_com_average.labels_
cluster_labels = pd.Series(tel_com_average.labels_)
tel_com['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

tel_com_centroid = AgglomerativeClustering(n_clusters = 3, linkage = 'centroid',affinity ="euclidean").fit(tel_com_norm)
tel_com_centroid.labels_
cluster_labels = pd.Series(tel_com_centroid.labels_)
tel_com['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

tel_com = tel_com.iloc[:,[28,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]
tel_com.head()

#Aggregate the mean of each cluster
tel_com.iloc[:,0:].groupby(tel_com.clust).mean()

#creating a csv file
tel_com.to_csv("new_tel_com.csv", encoding = "utf-8")
os.getcwd()

################## gower ###################
import gower
from scipy.cluster.hierarchy import fcluster , dendogram
gower_matrix = gower.gower_matric(tel_com)
gower_linkage = linkage(gowers_matrix)
gcluster = fcluster(gower_limkage , 3 , criterion = 'maxclust')
dendogram(gower_linkage)
tel_com["cluster"] = gcluster
tel_com.iloc[: , 0:29].groupy(tel_data.cluster).mean()
############################################ problem4 ##########################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import seaborn as sns
import numpy as np
import os

auto_ins = pd.read_csv("C:\\Users\\DELL\\Downloads\\AutoInsurance.csv")
auto_ins.describe()
auto_ins.info()
auto_ins.isna().sum()
auto_ins.columns

#to check duplicates
dups = auto_ins.duplicated()
sum(dups)

auto_ins = auto_ins.drop_duplicates()

#creat dummy variables on categorical data
auto_ins_new = pd.get_dummies(auto_ins)

#creating instance of one hot encoding
auto_ins = pd.read_csv("C:\\Users\\DELL\\Downloads\\AutoInsurance.csv")
auto_ins.describe()
auto_ins.drop(['Number of Open Complaints'],axis = 1,inplace = True)
auto_ins.dtypes

from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
one_hot_auto_ins = pd.DataFrame(one_hot.fit_transform(auto_ins).toarray())

#creating instance of labelencoder
auto_ins = pd.read_csv("C:\\Users\\DELL\\Downloads\\AutoInsurance.csv")
auto_ins.describe()
auto_ins.drop(['Number of Open Complaints'],axis = 1,inplace = True)
auto_ins.dtypes

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x = auto_ins.iloc[: , :]

x['Customer'] = label_encoder.fit_transform(x['Customer'])
x['State'] = label_encoder.fit_transform(x['State'])
x['Customer Lifetime Value'] = label_encoder.fit_transform(x['Customer Lifetime Value'])
x['Response'] = label_encoder.fit_transform(x['Response'])
x['Coverage'] = label_encoder.fit_transform(x['Coverage'])
x['Education'] = label_encoder.fit_transform(x['Education'])
x['Effective To Date'] = label_encoder.fit_transform(x['Effective To Date'])
x['EmploymentStatus'] = label_encoder.fit_transform(x['EmploymentStatus'])
x['Gender'] = label_encoder.fit_transform(x['Gender'])
x['Income'] = label_encoder.fit_transform(x['Income'])
x['Location Code'] = label_encoder.fit_transform(x['Location Code'])
x['Marital Status'] = label_encoder.fit_transform(x['Marital Status'])
x['Monthly Premium Auto'] = label_encoder.fit_transform(x['Monthly Premium Auto'])
x['Months Since Last Claim'] = label_encoder.fit_transform(x['Months Since Last Claim'])
x['Months Since Policy Inception'] = label_encoder.fit_transform(x['Months Since Policy Inception'])
x['Number of Open Complaints'] = label_encoder.fit_transform(x['Number of Open Complaints'])
x['Number of Policies'] = label_encoder.fit_transform(x['Number of Policies'])
x['Policy Type'] = label_encoder.fit_transform(x['Policy Type'])
x['Policy'] = label_encoder.fit_transform(x['Policy'])
x['Renew Offer Type'] = label_encoder.fit_transform(x['Renew Offer Type'])
x['Sales Channel'] = label_encoder.fit_transform(x['Sales Channel'])
x['Total Claim Amount'] = label_encoder.fit_transform(x['Total Claim Amount'])
x['Vehicle Class'] = label_encoder.fit_transform(x['Vehicle Class'])
x['Vehicle Class'] = label_encoder.fit_transform(x['Vehicle Class'])


#univariate analysis
#boxplot
plt.boxplot(auto_ins["Customer Lifetime Value"])
plt.boxplot(auto_ins["Income"])
plt.boxplot(auto_ins["Monthly Premium Auto"])
plt.boxplot(auto_ins["Months Since Last Claim"])
plt.boxplot(auto_ins["Months Since Policy Inception"])
plt.boxplot(auto_ins["Number of Open Complaints"])
plt.boxplot(auto_ins["Number of Policies"])
plt.boxplot(auto_ins["Total Claim Amount"])

#detection of outliers(find the RM based on IQR)
IQR = auto_ins["Customer Lifetime Value"].quantile(0.75) - auto_ins["Customer Lifetime Value"].quantile(0.25)
lower_limit_Number_of_Referrals = auto_ins["Customer Lifetime Value"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = auto_ins["Customer Lifetime Value"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
auto_ins["Customer Lifetime Value"] = pd.DataFrame(np.where(auto_ins["Customer Lifetime Value"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(auto_ins["Customer Lifetime Value"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,auto_ins["Customer Lifetime Value"])))
sns.boxplot(auto_ins["Customer Lifetime Value"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = auto_ins["Monthly Premium Auto"].quantile(0.75) - auto_ins["Monthly Premium Auto"].quantile(0.25)
lower_limit_Number_of_Referrals = auto_ins["Monthly Premium Auto"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = auto_ins["Monthly Premium Auto"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
auto_ins["Monthly Premium Auto"] = pd.DataFrame(np.where(auto_ins["Monthly Premium Auto"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(auto_ins["Monthly Premium Auto"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,auto_ins["Monthly Premium Auto"])))
sns.boxplot(auto_ins["Monthly Premium Auto"]);plt.title("Boxplot");plt.show()


#detection of outliers(find the RM based on IQR)
IQR = auto_ins["Total Claim Amount"].quantile(0.75) - auto_ins["Total Claim Amount"].quantile(0.25)
lower_limit_Number_of_Referrals = auto_ins["Total Claim Amount"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = auto_ins["Total Claim Amount"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
auto_ins["Total Claim Amount"] = pd.DataFrame(np.where(auto_ins["Total Claim Amount"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(auto_ins["Total Claim Amount"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,auto_ins["Total Claim Amount"])))
sns.boxplot(auto_ins["Total Claim Amount"]);plt.title("Boxplot");plt.show()

#bivariate analysis
#scatter plot
sns.scatterplot(auto_ins["Total Claim Amount"],auto_ins["Monthly Premium Auto"])
sns.scatterplot(auto_ins["Total Claim Amount"],auto_ins["Customer Lifetime Value"])

#normalisation function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

auto_ins_norm = norm_func(auto_ins_new)

#model building
#dendogram
auto_ins_new_linkage = linkage(auto_ins_norm,method = "complete",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using complete linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(auto_ins_new_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

auto_ins_new_linkage  = linkage(auto_ins_norm,method = "single",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(auto_ins_new_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


auto_ins_new_linkage = linkage(auto_ins_norm,method = "average",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using average linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(auto_ins_new_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

auto_ins_new_linkage = linkage(auto_ins_norm,method = "centroid",metric = "euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using centroid linkge');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(auto_ins_new_linkage, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

#now applying AgglomerativeClustering choosing 3 as clusters from the above dendogram
auto_ins_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete',affinity ="euclidean").fit(auto_ins_norm)
auto_ins_complete.labels_
cluster_labels = pd.Series(auto_ins_complete.labels_)
auto_ins['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

auto_ins_single = AgglomerativeClustering(n_clusters = 3, linkage = 'single',affinity ="euclidean").fit(auto_ins_norm)
auto_ins_single.labels_
cluster_labels = pd.Series(auto_ins_single.labels_)
auto_ins['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

auto_ins_average = AgglomerativeClustering(n_clusters = 3, linkage = 'average',affinity ="euclidean").fit(auto_ins_norm)
auto_ins_average.labels_
cluster_labels = pd.Series(auto_ins_average.labels_)
auto_ins['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

auto_ins_centroid = AgglomerativeClustering(n_clusters = 3, linkage = 'centroid',affinity ="euclidean").fit(auto_ins_norm)
auto_ins_centroid.labels_
cluster_labels = pd.Series(auto_ins_centroid.labels_)
auto_ins['clust'] = cluster_labels #creating a new coloumn and assigning it to a new coloumn

auto_ins = auto_ins.iloc[:,[23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
auto_ins.head()

#Aggregate the mean of each cluster
auto_ins.iloc[:,0:].groupby(auto_ins.clust).mean()

#creating a csv file
auto_ins.to_csv("new_auto_ins.csv", encoding = "utf-8")
os.getcwd()

















































