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
import sys
#Opening the correct folder
sys.path.append("C://Users/Zohaib/Desktop/Lectures/Udacity/Machine Learning/Machine Learning/tools")
from feature_format import featureFormat, targetFeatureSplit
#Loading the dataset
data_dict = pickle.load(open("C://Users/Zohaib/Desktop/Lectures/Udacity/Machine Learning/Machine Learning/final_project/final_project_dataset.pkl", "r") )

#List of features to use in the analysis        
features_list = ["poi", "exercised_stock_options","bonus","expenses"]
#Splitting the data using the feature format function
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
feat1=pd.DataFrame(features)
for i in feat1:
    scaler.fit(feat1[i].values.reshape(-1,1))
    feat1[i]=scaler.transform(feat1[i].values.reshape(-1,1))
features=np.array(feat1)
    
#Running 4 iterations of the training test split so as to encompass every combination of data points
kf = KFold(len(features),3,shuffle=True,random_state=1)
#Defining function to check estimators
def estmt(estm,prmt):
#        Setting the parameters we want to null
        acc={}
        recall={}
        precision={}
        f1={}
        ind=0
#        for every iteration of the KFold
        for train_index,test_index in kf:
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
#Run all of the classifiers and parameters through the function
dacc,drecall,dprecision,df1,dclf=estmt(dtc,dtcp)
gacc,grecall,gprecision,gf1,gclf=estmt(GNB,{'priors':[None]})
kacc,krecall,kprecision,kf1,kclf=estmt(KNC,kneigh)
kneighbest={'algorithm': 'auto', 'leaf_size': 5, 'n_neighbors': 5, 'weights': 'uniform'}

#Save relevant files in a pickle file
pickle.dump(data_dict,open("C://Users/Zohaib/Desktop/Lectures/Udacity/Machine Learning/Machine Learning/final_project/my_dataset.pkl",'w'))
pickle.dump(features_list,open("C://Users/Zohaib/Desktop/Lectures/Udacity/Machine Learning/Machine Learning/final_project/my_feature_list.pkl",'w'))
pickle.dump(KNeighborsClassifier(kneighbest),open("C://Users/Zohaib/Desktop/Lectures/Udacity/Machine Learning/Machine Learning/final_project/my_classifier.pkl",'w'))
