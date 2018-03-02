from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import pickle
import sys,os
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier


#loading data manipulating functions
from feature_format import featureFormat, targetFeatureSplit
#Loading the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#Getting a list ready for Selectkbest
flist1=['poi']
#Removes the Total value from the dataset since this value should not be used for identifying pois.
data_dict.pop("TOTAL")
#Adds poi email interaction into the dataset and creates feature list for select k best
for n,i in enumerate(data_dict):
    a=data_dict[i]['from_poi_to_this_person']
    b=data_dict[i]['from_this_person_to_poi']
    c=data_dict[i]['from_messages']
    d=data_dict[i]['to_messages']
    lst=[a,b,c,d]
    for valuen,value in enumerate(lst):
        if value=='NaN':
            lst[valuen]=0
    try:
        data_dict[i]['poi_email_interaction']=(lst[0]+lst[1])/(lst[2]+lst[3])
    except ZeroDivisionError:
        data_dict[i]['poi_email_interaction']='NaN'
for n,i in enumerate(data_dict):
    if n==0:
        for ftr in data_dict[i]:
            if ftr not in ['poi','email_address','other'] :
                flist1.append(ftr)
   
#Splitting the data using the feature format function and target feature split function
data = featureFormat(data_dict, flist1)
labels, features1 = targetFeatureSplit(data)

#Computes Anova Fscores for every feature in the dataset, and will return the 4 best
fvalue_selector = SelectKBest(f_classif, k=4)
fvalue_selector.fit(features1,labels)
features=fvalue_selector.transform(features1)
#List made by checking the scores
features_list=['poi','exercised_stock_options','total_stock_value','bonus','salary']
scores={}
for n,i in enumerate(flist1[1:]):
    scores[i]=list(fvalue_selector.scores_)[n]
    
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
feat1=pd.DataFrame(features)
for i in feat1:
    scaler.fit(feat1[i].values.reshape(-1,1))
    feat1[i]=scaler.transform(feat1[i].values.reshape(-1,1))
features=np.array(feat1)

#Running 3 iterations of the training test split so as to encompass every combination of data points
ss = StratifiedShuffleSplit(n_splits=3,test_size=.2,random_state=42)
#Defining function to check estimators
def calc_eval(estm,prmt):
#        Setting the parameters we want to null
        acc={}
        recall={}
        precision={}
        f1={}
        ind=0
#        for every iteration of the KFold
        for train_index,test_index in ss.split(features,labels):
#            Set the training and testing data
            features_train = [features[i] for i in train_index]
            features_test =  [features[i] for i in test_index]
            labels_train = [labels[i] for i in train_index]
            labels_test = [labels[i] for i in test_index]
#            Create the gridsearch object that will iterate through all combinations of parameters to get the best one
            clf=GridSearchCV(estm, prmt)
#            fit the training data to the estimator
            clf.fit(features_train,labels_train)
#            predict using the fit estimator
            pred=clf.predict(features_test)
#            calculate the relevant statistics
            acc[ind]=accuracy_score(labels_test,pred)
            recall[ind]=recall_score(labels_test,pred)
            precision[ind]=precision_score(labels_test,pred)
            f1[ind]=f1_score(labels_test,pred)
            ind+=1
#        calculate the average of the 4 iterations of the kfold
        acct=np.mean(acc.values())
        recallt=np.mean(recall.values())
        precisiont=np.mean(precision.values())
        f1t=np.mean(f1.values())
        return acct,recallt,precisiont,f1t,clf
#Decistion Tree parameters
dtcp={'criterion':('gini', 'entropy'),'min_samples_split':[2,10,20]}
#Decision tree classifier
dtc = DecisionTreeClassifier()
#Naive Bayes Classifier
GNB=GaussianNB()
#KNeighbors parameters
kneigh={'n_neighbors':[1,3,5,7,10],'weights':['uniform','distance'],'algorithm':['auto','ball_tree','kd_tree','brute'],'leaf_size':[5,10,20,30,100]}

#KNeighbors classifier
KNC=KNeighborsClassifier()

#Random Forest Classifier
rfc=RandomForestClassifier()

rfcp={'n_estimators':[1,5,10,20,100],'criterion':['gini','entropy'],'min_samples_split':[2,10,20]}
#Run all of the classifiers and parameters through the function
dacc,drecall,dprecision,df1,dclf=calc_eval(dtc,dtcp)
gacc,grecall,gprecision,gf1,gclf=calc_eval(GNB,{'priors':[None]})
kacc,krecall,kprecision,kf1,kclf=calc_eval(KNC,kneigh)
kneighbest={'algorithm': 'auto', 'leaf_size': 5, 'n_neighbors': 5, 'weights': 'uniform'}
dbest={'criterion': 'entropy', 'min_samples_split': 20}
#Save relevant files in a pickle file
pickle.dump(data_dict,open("my_dataset.pkl",'w'))
pickle.dump(features_list,open("my_feature_list.pkl",'w'))
pickle.dump(GaussianNB(),open("my_classifier.pkl",'w'))





