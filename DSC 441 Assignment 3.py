#!/usr/bin/env python
# coding: utf-8

# # DSC 441 Assignment 3
# 
# ## Name : Jonathan Sands
# ## Date: 10/22/2020

# In[2]:


#Operation and data manipulation modules
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats

#Machine Learning modules
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import mean_squared_log_error

#Visualization
import matplotlib.pyplot as plt


# In[143]:


#Retrieve data from it's storage location on my computer
data = pd.read_csv(r'D:\DePaul\DSC 441\sledata.txt', sep = " ", header = None)

#Assign numeric column names
data.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']


# In[144]:


#Split the data to an 80/20 train test split seperation
trainData, testData = model_selection.train_test_split(data, train_size = 0.8, random_state = 0)

#separate desired columns for train data
xTrain = trainData.iloc[:,:-1]
yTrain = trainData.iloc[:,-1:]


# In[145]:


#separate desired columns for the test data
yTest = testData.iloc[:,-1:]
xTest = testData.iloc[:,:-1]


# In[162]:


#Create the decision tree
clf = tree.DecisionTreeClassifier(criterion = "gini", min_samples_split = 2, min_samples_leaf = 2, min_impurity_decrease = .001, random_state = 0)

clf = clf.fit(xTrain, yTrain)

fig = plt.figure(figsize = (25, 25))
tree.plot_tree(clf, feature_names = xTrain.columns, filled = True)


# In[165]:


prediction = clf.predict(xTest)


#Finding the R Score 
print('The R squared score is :  {} \n\nThis tells us that the model fits our data well'.format(clf.score(xTest, yTest)))

print('\n\nOur error for this classifier was {}%'.format(mean_squared_log_error(yTest, prediction)*100))

print('\nThe total nodes for the tree are {}'.format(clf.tree_.node_count))

importance = clf.feature_importances_
print('\n')
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# ## Problem 1
# 
# ##### Above is the plot I created for the decision tree.
# 
# #### 1.)
# 
# The criterion used to measure the quality of the split in this classifier is the Gini impurity measure. This is the default, but the other supported type is entropy for information gain.
# 
# The minimum number of samples required to create a leaf are set to 2, this proved to give us the largest possible $ R^2 $ value along with the smallest error.
# 
# Their is no depth restriction for the tree, meaning that the nodes expand until all leaves are pure or until all leaves contain less than the minimum sample set, which is 2.
# 
# The error testing with the mean_squared_log_error() was just over 1%. 
# 
# #### 2.)
# 
# There are 13 leaves which in addition to the other nodes help make up the 29 total nodes
# 
# #### 3.)
# 
# The three most important features for making predictions are 9, 10, and 0 based on using the CART algorithm for feature importance and using the GINI as the criterion for splits.
# 
# The Gini equation is:
# 
# $ Gini = 1 - \sum_{n=1}^{n}p_j^2$
# 
# $Gini_s = \sum_{i = 1}^{k} \frac{n_i}{n}Gini(i)$
# 
# #### 4.)
# 
# As we continue to increase the min_sample_split, the number of nodes decreases, decreasing it's complexity. Our error and accuracy also increase gradually as we increase the minimum sample split.

# ***
# ###### Personal Notes
# The R-Score tells us that there our model fits the data well using the $ R^2 $ score as a measurement.
# The $ R^2 $ Score divides the  residual SumSquaredRegression (SSR) by the total sumn of square (SST):
# 
# 
# $ R^2 = \dfrac{RSS}{TSS} $
# 
# 
# $RSS = \sum_{i = 1}^{n}R_n - R $
# 
# 
# $TSS = \sum_{i = 1}^{n}T_n - T $
# 
# 
# Where $ R_n $ is the residual of sample $ n $. Similarly, $ T_n $ is the difference in a sample $ n $'s summable observations and the mean summable observations overall.
# ***
# 
# 
# 
# Sklearn uses the Gini Index to compute importance and does it for us automatically at each node using:
# 
# $ Gini = 1 - \sum_{n=1}^{n}p_j^2$
# 
# $Gini_s = \sum_{i = 1}^{k} \frac{n_i}{n}Gini(i)$
# 
# where $P_j $ is the probability of a tuple $ j $ belonging to a specific class.
# 

# ## Problem 2 (30 points)
# #### This problem illustrates the effect of the class imbalance of the accuracy of the decision
# trees. Download the red wine quality data from the UCI machine learning repository at:
# http://archive.ics.uci.edu/ml/datasets/Wine+Quality

# ###### 1.)

# In[180]:


#Collect Data 
data = pd.read_csv(r'C:\Users\Jonathan Sands\Downloads\winequality-red.csv', sep = ";")
print('There are 6 classes, those 6 classes are: {}'.format(data.quality.unique()))
data.head()


# In[181]:


sns.distplot(data.quality)


# Clearly the code is heavily normally distributed. Majority of the datapoints are in the 5 and 6 range.

# ###### 2.)

# In[168]:


#Building the classifier
Train, Test = model_selection.train_test_split(data, train_size = 0.8, random_state = 0)

xTrain = Train.iloc[:,:-1]
yTrain = Train.iloc[:,-1:]

xTest = Test.iloc[:,:-1]
yTest = Test.iloc[:,-1:]


# In[191]:


for i in xTrain.columns:
    
    xTrain[i] = stats.zscore(xTrain[i])

for i in xTest.columns:
    
    xTest[i] = stats.zscore(xTest[i])


# In[222]:


clf = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 50, min_samples_leaf = 2, random_state = 0)

clf = clf.fit(xTrain, yTrain)

fig = plt.figure(figsize = (200, 200));
tree.plot_tree(clf, feature_names = xTrain.columns, filled = True);


# In[224]:


prediction = clf.predict(xTest)
print('Our error for this classifier was {}%'.format(mean_squared_log_error(yTest, prediction)*100))

print('\nThe R squared score is :  {} \n\nThis tells us that the model fits our data well'.format(clf.score(xTest, yTest)))

print('\nThe total nodes for the tree are {}'.format(clf.tree_.node_count))

importance = clf.feature_importances_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# ### Problem 1 Questions for Problem 2
# ###### 1.) 
# 
# After playing around with it for a bit, I found that having a max depth of 10, with the minimum number of samples to cause a split = 50 and the minimum number of samples to make a node to be 2 gave me the most accurate predicitons and the best fitted model.
# 
# ###### 2.)
# 
# There are 93 nodes for the tree, and I believe 48 leaves, this may be incorrect as I hand counted the leaves but it appears to be about right. I feel as though there may be some overfitting as a result of this but the predictions are better with a larger amount of leaves. If we had more data this would be a thing I would test using a CV set.
# 
# ###### 3.)
# 
# The best features appear to be alcohol, sulphates, and volatile acidity. I got these results once again by using the CART algorithm for feature importance and using the GINI as the criterion for splits.
# 
# The Gini equation is:
# 
# $ Gini = 1 - \sum_{n=1}^{n}p_j^2$
# 
# $Gini_s = \sum_{i = 1}^{k} \frac{n_i}{n}Gini(i)$
# 
# A few others such as sulfer dioxide have some good moments in the tree in regards to their GINI, but too many bad to outweigh the true starts listed above.
# 
# ###### 4.)
# 
# This tree in it's most complex form is the best. When the number of samples in a node are set to the smallest min, it returns the most accurate fit. This may be caused by overfitting but since it also predicts the test data set the best I will leave it.

# In[3]:


#Reload the data to bin
data = pd.read_csv(r'C:\Users\Jonathan Sands\Downloads\winequality-red.csv', sep = ";")
data.head()


# In[6]:


#Find the average to do mean bins
LOW = []

FAIR = []

GOOD = []

HIGH = []

for i in range(len(data.quality)):
    
    if data.quality[i] <= 4:
        
        LOW.append(data.quality[i])
        
    elif data.quality[i] == 5:
        
        FAIR.append(data.quality[i])
        
    elif data.quality[i] == 6:
        
        GOOD.append(data.quality[i])
        
    elif data.quality[i] >= 7:
        
        HIGH.append(data.quality[i])
        

mean_LOW = sum(LOW)/len(LOW)

mean_FAIR = sum(FAIR)/len(FAIR)

mean_GOOD = sum(GOOD)/len(GOOD)

mean_HIGH = sum(HIGH)/len(HIGH)


# In[8]:


#Create the bins
for i in range(len(data.quality)):
    
    if data.quality[i] <= 4:
        
        data.quality[i] = mean_LOW
        
    elif data.quality[i] == 5:
        
        data.quality[i] = mean_FAIR
        
    elif data.quality[i] == 6:
        
        data.quality[i] = mean_GOOD
        
    elif data.quality[i] >= 7:
        
        data.quality[i] = mean_HIGH


# In[9]:


#Building the classifier
Train, Test = model_selection.train_test_split(data, train_size = 0.8, random_state = 0)

xTrain = Train.iloc[:,:-1]
yTrain = Train.iloc[:,-1:]

xTest = Test.iloc[:,:-1]
yTest = Test.iloc[:,-1:]


# In[10]:


clf = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 50, min_samples_leaf = 2, random_state = 0)

clf = clf.fit(xTrain, yTrain)

fig = plt.figure(figsize = (200,200))
tree.plot_tree(clf, feature_names = xTrain.columns, filled = True);


# In[11]:


prediction = clf.predict(xTest)
print('Our error for this classifier was {}%'.format(mean_squared_log_error(yTest, prediction)*100))

print('\nThe R squared score is :  {} \n\nThis tells us that the model fits our data well'.format(clf.score(xTest, yTest)))

print('\nThe total nodes for the tree are {}'.format(clf.tree_.node_count))

importance = clf.feature_importances_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# ### Problem 1 Questions for binning
# 
# ###### 1.)
# 
# I set the minimum number of samples required for a node to be a leaf to 2, the min required for a split to 50 and the max depth to 10 because it led the data to match the model better and minimized the error to 3% 
# 
# ###### 2.)
# 
# There are 29 terminal nodes which add with the other nodes to create 59 total nodes.
# 
# ###### 3.)
# 
# Alcohol (Feature 0: 0.10), sulphates (Feature 1: 0.17219), and total sulfur dioxide (Feature 6: 0.09946) appear to be the strongest features this time using the CART algorithm. Volatile acidity (Feature 1: 0.09544) is important, just slightly less than the others.
# 
# Sulfer dioxides and pH also appear to be a valuable feature.
# 
# ###### 4.)
# 
# Complexity definitely decreases as the minimum split is increased because there are more samples allowed in terminal nodes.

# ### Problem 2 
# 
# ###### Question 4
# 
# Binning makes the classes more balanced. It can improve the performance of the model and/or it can decrease the complexity of the model in terms of the number of nodes. In the case above the complexity of the model decreased while the accuracy is the same.
# 
# ###### Question 5
# 
# If we began using cost sensitive approaches in order to avoid the loss of info through binning. If we did this we could potentially get better results in terms of lower and higher scores than binning provides. Over and undersampling the majority classes would also help.
# 
# Personally I would like to continue but just receive more data. I don't want to sacrifice any of my train or test data for cross validation, even though I know there are methods for a "take and replace" approach.

# ### Problem 3

# | Example | Color| Height | Width | Class|
# | --- | --- | --- | --- | --- | 
# | A | Red | Short | Thin | No | 
# | B | Blue | Tall | Fat | Yes |
# | C | Green | Short | Fat | No |
# | D | Green | Tall | Thin | Yes|
# | E | Blue | Short | Thin | No |
# 

# For the first example A, we check to see the color. Because the color is Red it's an automatic No
# 
# For the second example B, we check to see the color Blue, then we check width fat and we get Yes
# 
# For the third example C, we check to see the color Green, height in short so it's a No
# 
# For the fourth example D, we check to see the color Green, height is tall so it's a Yes
# 
# For the last example E, we check to see the color is Blue, width is thin so it's a No
# 
# 
