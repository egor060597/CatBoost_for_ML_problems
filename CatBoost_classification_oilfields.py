#!/usr/bin/env python
# coding: utf-8

# In[30]:


#prediction of the oilfields type (offshore/onshore) with CatBoost optimized by Optuna. Task from 2020 first round "Я профессионал"

#importing all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.preprocessing import label_binarize    
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split


# In[31]:


train = pd.read_csv('input_train.csv')


# In[32]:


train


# In[33]:


#preparing the data
cols = list(train.columns.values)

train = train[['Tectonic regime',
 'Onshore/Offshore',
 'Hydrocarbon type',
 'Reservoir status',
 'Structural setting',
 'Period',
 'Lithology',
 'Depth',
 'Gross',
 'Netpay',
 'Porosity',
 'Permeability']]

cat_features = ['Tectonic regime',
 'Hydrocarbon type',
 'Reservoir status',
 'Structural setting',
 'Period',
 'Lithology']

num_features = ['Depth',
 'Gross',
 'Netpay',
 'Porosity',
 'Permeability']

target = ['Onshore/Offshore']


# In[34]:


#encoding the data
y = train[target]
en = LabelEncoder()
y = en.fit_transform(y)
y = pd.DataFrame(y,columns = ['label'])
X = train[cat_features + num_features]


# In[35]:


# splitting the data into training and test set
X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
temp = pd.concat([X_t,y_t],axis = 1)
temp = temp.reset_index()
temp.drop(['index'],axis = 1,inplace = True)
X_t = temp[cat_features + num_features]
y_t = temp['label']


# In[36]:


temp


# In[37]:


# optimizing CatBoost by optuna on training and cross-validation dataset
import optuna

def objective(trial):
    

    param = {
        "iterations" : 3000,
        "learning_rate":trial.suggest_float("learning_rate",1e-06,1),
        "loss_function":"MultiClass",
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
    }
    
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        
    clf = CatBoostClassifier(**param)
    
    kf = KFold(n_splits = 5)
    i = 0
    roc_auc = []
    
    
    for train_index, test_index in kf.split(X_t):
        X_train, X_valid = X_t.loc[train_index], X_t.loc[test_index]
        y_train, y_valid = y_t.loc[train_index], y_t.loc[test_index]

 
        if i == 0:
            clf.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100,
                    cat_features=cat_features, verbose=False)
            clf.save_model('catboost', format="cbm", export_parameters=None, pool=None)
            preds = clf.predict(X_valid)
            preds = label_binarize(preds, classes=[0, 1, 2])
            y_valid_ = label_binarize(y_valid, classes=[0, 1, 2])
            fpr, tpr, thresholds = roc_curve(y_valid_.ravel(), preds.ravel())
            roc_auc.append(auc(fpr, tpr))
        else:
            clf.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100,
                    cat_features=cat_features, verbose=False, init_model='catboost')
            clf.save_model('catboost', format="cbm", export_parameters=None, pool=None)
            preds = clf.predict(X_valid)
            preds = label_binarize(preds, classes=[0, 1, 2])
            y_valid_ = label_binarize(y_valid, classes=[0, 1, 2])
            fpr, tpr, thresholds = roc_curve(y_valid_.ravel(), preds.ravel())
            roc_auc.append(auc(fpr, tpr))
    
    test_preds = clf.predict(X_test)
    test_preds_ = label_binarize(test_preds, classes=[0, 1, 2])
    y_test_ = label_binarize(y_test, classes=[0, 1, 2])
    fpr, tpr, thresholds = roc_curve(y_test_.ravel(), test_preds_.ravel())
    
    accuracy = auc(fpr,tpr)
    
    return accuracy


# In[38]:


study = optuna.create_study(direction = "maximize")

study.optimize(objective,n_trials = 5)


# In[39]:


#exctracting the best parameters
study.best_params


# In[43]:


#inserting the best set to CatBoost
parameters = {'learning_rate': 0.7883836457987905,
 'colsample_bylevel': 0.05016922938088764,
 'depth': 5,
 'boosting_type': 'Plain',
 'bootstrap_type': 'MVS'}

control = CatBoostClassifier(**parameters)


# In[44]:


#preparing test set data
test_data = pd.read_csv('input_test.csv')
test = test_data[cat_features + num_features]


# In[45]:


#training the model 
kf = KFold(n_splits = 5)
i = 0
roc_auc = []
    
    
for train_index, test_index in kf.split(X):
    X_train, X_valid = X.loc[train_index], X.loc[test_index]
    y_train, y_valid = y.loc[train_index], y.loc[test_index]

 
    if i == 0:
        control.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100,
                    cat_features=cat_features, verbose=False)
        control.save_model('catboost', format="cbm", export_parameters=None, pool=None)
        preds = control.predict(X_valid)
        preds = label_binarize(preds, classes=[0, 1, 2])
        y_valid_ = label_binarize(y_valid, classes=[0, 1, 2])
        fpr, tpr, thresholds = roc_curve(y_valid_.ravel(), preds.ravel())
    else:
        control.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100,
                    cat_features=cat_features, verbose=False, init_model='catboost')
        control.save_model('catboost', format="cbm", export_parameters=None, pool=None)
        preds = control.predict(X_valid)
        preds = label_binarize(preds, classes=[0, 1, 2])
        y_valid_ = label_binarize(y_valid, classes=[0, 1, 2])
        fpr, tpr, thresholds = roc_curve(y_valid_.ravel(), preds.ravel())
    print(f"accuracy: {auc(fpr,tpr):{10}.2f}")
    print("---------------------------------------------")   
    


# In[46]:


# getting the predictions
test_preds = control.predict(test)


# In[47]:


#transforming back to normal values
test_preds = en.inverse_transform(test_preds.ravel())


# In[48]:


test_preds

