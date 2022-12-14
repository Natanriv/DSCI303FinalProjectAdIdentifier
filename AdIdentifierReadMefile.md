# DSCI303FinalProjectAdIdentifier
Ad Identifier Project


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import collections
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


# Check and Clean up Data


--- see data:

df=pd.read_csv("add.csv")
df.head()


#drop the index column:
df=df.drop('Unnamed: 0',axis=1)

#convert columns to integers:
df.columns=df.columns.astype('int')



df.dtypes.head(3)


#data set information:

print(df.info()) 



#This result should be float but is an object cause of the missing values
df.iloc[:,0:3].info()



#missing data how it should look:
df[0][10] 




#heatmap for missing data in the first 4 columns:


newdf=df.iloc[:,[0,1,2,3]]
newdf=newdf.applymap(lambda x:'?' in str(x))
plt.figure(figsize=(17,5))
plt.title("Missing values HeatMap")
print(sns.heatmap(newdf,cbar=False,yticklabels=False,cmap='viridis'))


#frequency of missing values:

for i in (newdf):
    print('column['+str(i)+'] has missing values -'+str(sum(newdf[i])))


#replace missing values with mean function

def replace_missing(df):
    """ This function will replace any miissing value with the aggregated mean of the data as a 'float' to increase precision """
    for i in df:
        df[i]=df[i].replace('[?]',np.NAN,regex=True).astype('float')
        df[i]=df[i].fillna(df[i].mean())
    return df
    
    
#Here we use the function above to facilitate replacing values:


df[[0,1,2,3]]=replace_missing(df.iloc[:,[0,1,2,3]].copy()).values


#We use a lambda function to convert float values into rounded decimals to get a nominal output:
df[3]=df[3].apply(lambda x:round(x))

#Double check the information of the dataset:
df.iloc[:,3:4].info() 


# Exploratory Data Analysis

#General information on the dataset:
df[[0,1,2,3]].describe()


#With this plot, we can visualize the data and see that it is right-skewed:
fig,ax=plt.subplots(nrows=1,ncols=3)
fig.set_figheight(5)
fig.set_figwidth(13)
sns.distplot(df[0],ax=ax[0])
sns.distplot(df[1],ax=ax[1])
sns.distplot(df[2],ax=ax[2])



#Simple pairplot:
sns.pairplot(data=df.iloc[:,[0,1,2,3]])


# Plot Continous Data

#here we will plot the continous data vs the ad and non ad to see how is it distributed:

fig,ax=plt.subplots(nrows=3,ncols=1)
fig.set_figheight(15)
fig.set_figwidth(18)
sns.stripplot(y=1558,x=0,data=df,ax=ax[0])
sns.stripplot(y=1558,x=1,data=df,ax=ax[1])
sns.stripplot(y=1558,x=2,data=df,ax=ax[2])
plt.show()

#Now, we decide to use a Boxplot to help us visualize / 
#understand the data. We can see that there is a very high number of outliers in the non-ad target, 
#the non-ad data is highly skewed, and the width of ad is much larger compared to nonad:

plt.figure(figsize=(15,10))
sns.boxplot(x=1558,y=1,data=df)
plt.xlabel('label-add/non-ad')
plt.ylabel('width')
plt.title("Boxplot:  Target Vs Width ")
plt.show()


# Change last column to numeric


df.iloc[:,-1]=df.iloc[:,-1].replace(['ad.','nonad.'],[1,0])



# Prepare Features


x = df.iloc[:,:-1]
y = df.iloc[:,-1]



# Scaling Data


scaled = StandardScaler()
x = scaled.fit_transform(x)


sns.pairplot(data=df.iloc[:,[0,1,2,-1]], hue=1558)
plt.show()



# Split Data

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.30,random_state=0) # 70% to train


# Modeling


#We will start checking which model is better for our data. 
#We defined two functions that help us define which model is better. 
#The first function will just take in the models that we want and fit the data passed in for that specific type of model. 
#The second function will use take in the models that are now fit with the data from the first function and generate a classification report. 
#This allows us to easily see their respected accuracy. With these two functions, we have a clean and streamlined way of checking for accuracy and 
#selecting which models we should use. 
#Additionally, if we see that a model is not suited for this data, we can easily replace it with another and see how the new one performs. 



#Fitting models
#this first function fit multiple models by sklearn and return the dictionary with values as  objects of models
def fit_models(classifiers,xtrain,ytrain):
    models=collections.OrderedDict()
    for constructor in classifiers:
        obj=constructor()
        obj.fit(xtrain,ytrain)
        models[str(constructor).split(':')[0]]=obj
    return models

#Classification Reports
#This function generate classification accuracy report for given input model objects
def classification_multi_report(ytest,models_array):
    for i in models_array:
        print('__________________________________________________')
        print('the model - '+str(i))
        print(classification_report(ytest,models_array[i].predict(xtest)))

#Cross Validation
#This function return cross validated accuray and the variance of given input model obejects
def cross_Fucntion(models,cv):
    accuracy={}
    for model in models:
        cross_val_array=cross_val_score(models[model],xtrain,ytrain,scoring='accuracy',cv=cv)
        accuracy[model]=[np.mean(cross_val_array),np.std(cross_val_array)]
    return accuracy

#This function calculate the grid search parameters and accuracy  for given input modles and return dictionary with each tupple containing accuracy and best parameters
def multi_grid_search(param_grid_array,estimator_list,x,y):
    d={}
    count=0
    for i in estimator_list:
        gc=GridSearchCV(estimator=estimator_list[i],param_grid=param_grid_array[count],scoring ='accuracy',cv=5).fit(x,y)
        d[i]=(gc.best_params_,gc.best_score_)
        count+=1
    return d



# Naive Bayes
    
#Naive bayes classifier
#mixed NB for continuous and binary features


#calculate continous variables predicted probbability with gaussianNb
gaussian=GaussianNB()
gaussian.fit(xtrain[:,0:4],ytrain)

#calculate binaru variables predicted probbability with bernoulliNB
bernoulli=BernoulliNB()
bernoulli.fit(xtrain[:,4:1558],ytrain)

#multiply each probaility and divide by prior probability which for the binary case of ours is 1

final_y= gaussian.predict_proba(xtest[:,0:4])*bernoulli.predict_proba(xtest[:,4:1558])


#final y is the predicted y which if the probability is higher we pick that label
for i in range(final_y.shape[0]):
  if final_y[i,0]<final_y[i,1]:
    final_y[i,:]= int(1)
  else:
    final_y[i,:]= int(0)


# Models 

print(classification_report(np.array(ytest), np.array(final_y[:,0])))

# In this section, we are creating a list of the models that we will test, fitting the data into each model, 
# then creating a classification report for each one to compare performances

classifiers=[SVC, KNeighborsClassifier, GaussianNB, RandomForestClassifier, LogisticRegression]

model_list=fit_models(classifiers,xtrain,ytrain)

classification_multi_report(ytest,model_list)


#This function will generate the average error rate for a given input
def average_error_rate(test, pred):
    diff = np.abs(test - pred)
    return np.divide(diff, test).mean()
    
    
#Here we are just going through each model we chose and calculing the training / test mean absolute error
#and the training / test mean error

for i in model_list:
        print('__________________________________________________')
        print('the model - '+str(i))
        y_pred_train_af = model_list[i].predict(xtrain)
        y_pred_test_af = model_list[i].predict(xtest)
        print("Training mean absolute error is: ", mean_absolute_error(ytrain, y_pred_train_af))
        print("Test mean absolute error is: ", mean_absolute_error(ytest, y_pred_test_af))
        print()
        print("Training mean error rate is: ", average_error_rate(ytrain, y_pred_train_af))
        print("Test mean error rate is: ", average_error_rate(ytest, y_pred_test_af))
        

# Bias and Variance Tradeoff       
        
#We will use the function that was written above for 20-fold cross validation
obj=cross_Fucntion(model_list,cv=20)
for model in obj:
    print('the model -'+str(model)+'has \n || crosss validated accuracy as  -> '+str(obj[model][0])+' | variance - '+str(obj[model][1])+' ||' )
    print('______________________________________________________________________________________________________________')




# Hyperparameter Optimization


#Here, we are creating different grids of parameters for the different models that we use that we will 
#pass into the Grid Search CV to optimize the hyperparameters.

param_grid_svm=[
    {
        'kernel':['linear'],'random_state':[0]
    },
     {
        'kernel':['rbf'],'random_state':[0]
     },
    
    {
        'kernel':['poly'],'degree':[1,2,3,4],'random_state':[0]
    }
]

param_grid_knn=[

    {   
        'n_neighbors':np.arange(1,50),
        'p':[2]
        
    }
]

param_grid_nb=[
    {}
]

param_grid_forest = [
  { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
  }
]


param_grid_logistic = [
  {
    'C': np.logspace(-3,3,7),
    'penalty': ['l1', 'l2']
  }
]



param_grid_array=[param_grid_svm, param_grid_knn, param_grid_nb, param_grid_forest, param_grid_logistic]
multi_grid_search(param_grid_array,model_list,xtrain,ytrain)



# Results



#The Logistic Regression Model was one of the best performing models after we did hypterparameter optimization.
#We chose this model as one of the two that we saw to perform best with our data.
#Here, we are showing the results. with this model.
classifier_logistic=LogisticRegression(C= 0.1, penalty= 'l2')

#Fit the data
classifier_logistic.fit(xtrain,ytrain)

#Create a visual Confusion Matrix that shows how the model performed
sns.heatmap(pd.crosstab(ytest,classifier_logistic.predict(xtest)),cmap='coolwarm')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

#Show the classification report once more to have those scores again
print(classification_report(ytest,classifier_logistic.predict(xtest)))



#Similarly to what we did above, we chose a Random Forest Classifier model to be our second best model based on
#its performance and the score from cross-validation.
classifier_forest=RandomForestClassifier()

#Fit the data
classifier_forest.fit(xtrain,ytrain)

#Create a visual Confusion Matrix that shows how the model performed
sns.heatmap(pd.crosstab(ytest,classifier_forest.predict(xtest)),cmap='coolwarm')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

#Show the classification report once more to have those scores again
print(classification_report(ytest,classifier_forest.predict(xtest)))


# Feature Selection

#Install mlxtend
!pip install mlxtend
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib



#Import sequenctial features selector as SFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


sfs = SFS(LogisticRegression(), 
           k_features=10,# the more features we want, the longer it will take to run
           forward=True, 
           floating=False, # see the docs for more details in this parameter
           verbose=0, # this indicates how much to print out intermediate steps
           scoring='neg_mean_absolute_error',
           cv=3)

sfs = sfs.fit(xtrain, ytrain)


#Here, we are visualizing the performance of the model with a given number of inputs to help see what would be 
#the best number of features.

fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

