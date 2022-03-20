# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:01:12 2022

@author: Dayanand
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import figure
import numpy as np
import seaborn as sns

# changing the directory
os.getcwd()
os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\dsp1\\DataSets")

# loading file

rawData=pd.read_csv("train_blackfriday_sales.csv")
predictionData=pd.read_csv("prediction_blackfirday_sales.csv")

rawData.shape
predictionData.shape

rawData.columns
predictionData.columns

# Dep var Purchase  column is not there in prediction data sets

# adding "Purchase" columns
predictionData["Purchase"]=0
predictionData.shape

rawData.info()
predictionData.info()

# we have 7 numerical vars and 5 categorical variables
# Age is also given as categorical variables

# divide datasets into train & test

from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawData,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

# adding source coulmns in all three datasets
trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionData["Source"]="Prediction"

# combining all three data sets
fullDf=pd.concat([trainDf,testDf,predictionData],axis=0)
fullDf.shape
fullDf.columns

# drop  User_ID & Product_ID column from datasets
fullDf.drop(["User_ID","Product_ID"],axis=1,inplace=True)
fullDf.shape

# Univariate Analysis

# Missing Value treatment

fullDf.isna().sum()
# we have NULL values in Product_Category_2 & Product_Category_3
fullDf.info()
# Product_Category_2 & Product_Category_3 are float data types

# Product_Category_2
tempMedian=fullDf.loc[fullDf["Source"]=="Train","Product_Category_2"].median()
tempMedian
fullDf["Product_Category_2"].fillna(tempMedian,inplace=True)
fullDf["Product_Category_2"].isna().sum()

#Product_Category_3
tempMedian=fullDf.loc[fullDf["Source"]=="Train","Product_Category_3"].median()
tempMedian
fullDf["Product_Category_3"].fillna(tempMedian,inplace=True)
fullDf["Product_Category_3"].isna().sum()

# Convert categorical variable into dummy variable
fullDf2=pd.get_dummies(fullDf,drop_first=True)
fullDf2.shape

#Sample databases
train=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test"],axis=1)
train.shape
test=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test"],axis=1)
test.shape
prediction=fullDf2[(fullDf2["Source_Train"]==0) & (fullDf2["Source_Test"]==0)].drop(["Source_Train","Source_Test"],axis=1)
prediction.shape

# Divide datasets into dep. and indep. variable
trainX=train.drop(["Purchase"],axis=1)
trainX.shape
trainY=train["Purchase"]
trainY.shape
testX=test.drop(["Purchase"],axis=1)
testY=test["Purchase"]
predictionX=prediction.drop(["Purchase"],axis=1)


# standardizing the dataset

from sklearn.preprocessing import StandardScaler
train_sampling=StandardScaler().fit(trainX)
trainXStd=train_sampling.transform(trainX)
testXStd=train_sampling.transform(testX)
predictionXStd=train_sampling.transform(predictionX)

trainXStd=pd.DataFrame(trainXStd,columns=trainX.columns)
testXStd=pd.DataFrame(testXStd,columns=testX.columns)
predictionXStd=pd.DataFrame(predictionXStd,columns=predictionX.columns)

### PCA
from sklearn.decomposition import PCA
PCA_Model=PCA(n_components=0.90).fit(trainXStd)


# Len of PCA_MODEL
len(PCA_Model.components_)

# co efficients of PC variables-eigrnvectors
PCA_Model.components_

# Converting into dataframe

PC_DF=pd.DataFrame(PCA_Model.components_)
PC_DF

PC_DF_T=PC_DF.transpose()
PC_DF_T

PC_DF_T.index=trainXStd.columns
PC_DF_T

# Variation ratio # eigen values
print(PCA_Model.explained_variance_ratio_)

print(sum(PCA_Model.explained_variance_ratio_))

# Transform

trainXtransform=pd.DataFrame(PCA_Model.transform(trainXStd))
trainXtransform.shape
testXtransform=pd.DataFrame(PCA_Model.transform(testXStd))
testXtransform.shape

# Add constant

from statsmodels.api import add_constant
trainX=add_constant(trainXtransform)
trainX.shape
testX=add_constant(testXtransform)
#predictionX=add_constant(predictionX[sigCol])


## Linear Regression Model on PC's significant variables

trainX.head()
from statsmodels.api import OLS
PC_Linear_Model=OLS(np.log(trainY.reset_index(drop=True)),trainX).fit()
PC_Linear_Model.summary()

# Selecting significant Variables

maxPvalues=0.05
maxCutoff=0.05
trainXCopy=trainX.copy()
highPValuescolumns=[]

while(maxPvalues>=maxCutoff):
    
    tempDf=pd.DataFrame()
    tempModel=OLS(np.log(trainY.reset_index(drop=True)),trainXCopy).fit()
    tempDf["PValues"]=tempModel.pvalues
    tempDf["Columns"]=trainXCopy.columns
    
    maxPvalues=tempDf.sort_values(["PValues"],ascending=False).iloc[0,0]
    tempColumnName=tempDf.sort_values(["PValues"],ascending=False).iloc[0,1]
    
    if(maxPvalues>=maxCutoff):
        trainXCopy=trainXCopy.drop(tempColumnName,axis=1)
        highPValuescolumns.append(tempColumnName)
highPValuescolumns

trainX=trainX.drop(highPValuescolumns,axis=1)
testX=testX.drop(highPValuescolumns,axis=1)
#predictionX=predictionX.drop(highPValuescolumns,axis=1)

trainX.shape
testX.shape
#predictionX.shape

### Final Model
Model=OLS(np.log(trainY.reset_index(drop=True)),trainX).fit()
Model.summary()

# Prediction on Test Data
Test_Predict=np.exp(Model.predict(testX))
Test_Predict.head()

# RMSE
np.sqrt(np.mean((testY-Test_Predict)**2))

# MAPE
(np.mean(abs((testY-Test_Predict)/testY)))*100
