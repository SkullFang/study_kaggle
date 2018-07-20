#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/20/2018 2:30 PM
# @Author  : SkullFang
# @Contact : yzhang.private@gmail.com
# @File    : bagging.py
# @Software: PyCharm
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
#data processing
#miss age
titanic_data=pd.read_csv("./data/train.csv")
titanic_test=pd.read_csv("./data/test.csv")
def set_miss_age(df):
    #median fill nan
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Fare']=df['Fare'].fillna(df['Fare'].median())
    return df

def set_cabin_type(df):
    df.loc[(df['Cabin'].notnull()),'Cabin']='Y' #loc is read all rows
    df.loc[(df['Cabin'].isnull()), 'Cabin'] = 'N'  # loc is read all rows
    return df

def bulid_scaler(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))


def standard_age_fare(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)
    return df

def drop_no_need(df):
    return df.drop(['PassengerId','SibSp','Parch','Age','Fare'],axis=1)

def trans_to_num(df):
    dummies_Cabin=pd.get_dummies(df["Cabin"],prefix="Cabin")
    dummies_Embarked = pd.get_dummies(df["Embarked"], prefix="Embarked")
    dummies_Sex = pd.get_dummies(df["Sex"], prefix="Sex")
    dummies_Pclass = pd.get_dummies(df["Pclass"], prefix="Pclass")
#     #cat new series
    cat_df=pd.concat([df,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
    result_df=cat_df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    return result_df

def data_processing_pipeline(rude_data):
    set_miss_age(rude_data)
    set_cabin_type(rude_data)
    train_data = trans_to_num(rude_data)
    # print(train_data)

    return train_data


def build_model(ndarray):
    pass


train_data=data_processing_pipeline(rude_data=titanic_data)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(train_data['Age'].values.reshape(-1, 1))
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(train_data['Fare'].values.reshape(-1, 1))
train_data['Fare_scaled'] = scaler.fit_transform(train_data['Fare'].values.reshape(-1, 1), fare_scale_param)
train_data=drop_no_need(train_data).values
print(train_data.shape)
y_label=train_data[:,0]
X=train_data[:,1:]
clf=linear_model.LogisticRegression(C=0.1,penalty='l1',tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y_label)



#test
test_data=data_processing_pipeline(rude_data=titanic_test)

test_data['Age_scaled'] = scaler.fit_transform(test_data['Age'].values.reshape(-1, 1), age_scale_param)
test_data['Fare_scaled'] = scaler.fit_transform(test_data['Fare'].values.reshape(-1, 1), fare_scale_param)
test=drop_no_need(test_data).values
print(test.shape)
predictions=bagging_clf.predict(test)
result=pd.DataFrame({'PassengerId':test_data['PassengerId'].values,
                     'Survived':predictions.astype(np.int32)})
result.to_csv('./result/baggingmodel.csv',index=False)

# print(X[:,0])
