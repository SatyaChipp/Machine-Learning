"""
ML python sklearn package
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
"""

import numpy
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import urllib
# import urllib2
import requests

# import Image


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pd.read_csv(url, names=names)
##save file locally
resp1 = urllib.request.urlopen(url)
# resp2 = urllib2.urlopen(url)
resp3 = requests.get(url).text
print(dataset.head())
print(type(resp1.read()))  # bytes
print(type(resp3))  # string
##write to file
# with open('./local_dataset.csv', "a") as fi: # use 'x' to return error if file exists
#     fi.write(resp3)
# #read from file
# with open('./local_dataset.csv', 'r') as fi:
#     for line in fi:
#         print(str(line))

print(dataset.shape)
print(dataset.describe())
print(dataset.groupby('class').size())  ##each class has the same number of instances
###DATA VISUALISATION to bette understand our attributes
# box and whisker plot -- univariate plots first
# box plots display five-number summary of a set of data: minimum, mfirst quartile, median, thrid qurtile, maximum
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False)
# plt.show()
plt.plot()
plt.savefig('./image1.jpg')
dataset.hist()  # we see that some of the input variables have gaussian distribution
plt.plot()
plt.savefig('./image2.jpg')  ##conda install pillow  -  for this to work
##multivariate plots for more interactions between the variables
##scatter plot matrix: correlation between variables
scatter_matrix(dataset)
# plt.show()
plt.plot()
plt.savefig('./image3.jpg')

######EVAULATE ALGORITHMS
##now that the data prep is done , create some models and estimate their accuracy
# 1.make the validation set
# 2.setup test harness to use 10-fold cross validation
# 3. build 5 diff models to predict species from flower measurements
# 4. select the best model
###1
# split out validation dataset
array = dataset.values
# print(type(array))
# print(dataset.values)
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = \
    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

######Test Harness
##using 10 fold cross validation to estimate accuracy
##this splits our dataset into 10, train 9 and test on 1 and repeat for all combinations of train-test splits
seed = 7
scoring = 'accuracy' #we are using 'accuracy' metric to evaulate models
##accuracy -  ratio of the number of correctly predicted instances by total number of instances in dataset

######Build models
#from the plots we see that some of the classes are linearly separable in some dimensions
# algos we can evaulate:
# Logistic Regression, linear discriminatn analysis, k nearest neighbors, classification and regression trees, gaussian naive bayes, support vector machines
#LR, LDA,KNN,CART,NB,SVM
##reset the random seed
#spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier())) ##classification and regression trees

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

#evaluate each model now
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed) #for cross validation

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    # print(cv_results)
    names.append(name)
    mesg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(mesg)
    ###find that SVM has highest accuracy (highest mean and lowest std)
##each algo was evaulated ten times
#compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm comparision')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.plot()
plt.savefig('./imageAccuracies.jpg') #svm and knn have similar sdists for dataset
#####MAke predictions
#use knn to make final predictions and summarize results as final accuracy score

#make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(predictions)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

svm = SVC(gamma='auto')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(predictions)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))