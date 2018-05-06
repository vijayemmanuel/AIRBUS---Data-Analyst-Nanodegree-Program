#!/usr/bin/python

import sys
import pickle
from numpy import average
from numpy import median
from time import time
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#Utility functions

def extract_feature(data, feature_name):
    result = {}
    for item in data:
        if data[item].has_key(feature_name):
            result[item] = data[item][feature_name]
    return result

def add_feature(data, feature_name):
    for item in data:
        data[item][feature_name] = 0
    return data 

def remove_data(data, pois):
    result = {}
    for item in data:
        if item not in pois:
            result[item]= data[item]
    return result


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print  'Available Features', data_dict.values()[0].keys()


print data_dict.values()[0]

# Find the total number of POI
print 'Total poi', len(extract_feature(data_dict,'salary') )  


# Display keys 
print data_dict.keys()

#Zombie man
print data_dict['LOCKHART EUGENE E']
data_dict = remove_data(data_dict,['LOCKHART EUGENE E'])
data_dict = remove_data(data_dict,['THE TRAVEL AGENCY IN THE PARK'])

### Task 2: Remove outliers
# filter out the POI with no salary provided
poi_with_no_salary =  { k:v for k,v in extract_feature(data_dict,'salary').iteritems() if v == 'NaN'}
print 'Total poi after removal of salary with NaN', len(poi_with_no_salary)

data_dict = remove_data(data_dict,poi_with_no_salary.keys())

print 'Average Salary' , average(extract_feature(data_dict,'salary').values())
print 'Median Salary', median(extract_feature(data_dict,'salary').values())

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()


# Here you go, the culprit outlier which is called TOTAL
print 'Culprit Outlier' , [k for k,v in extract_feature(data_dict,'salary').iteritems() if v == max(extract_feature(data_dict,'salary').values())]
# Remove TOTAL
data_dict = remove_data(data_dict,['TOTAL'])


data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()



### Task 3: Create new feature(s)
## New Feature: Total Stock Value  = 'Exercised Stock Options' + 'Restricted Stock' + 'Restricted Deferred Stock'
add_feature(data_dict,'total_stock_value')
for item in data_dict:
    restricted_stock = 0
    exercised_stock = 0
    restricted_deferred_stock = 0
    if data_dict[item]['restricted_stock'] != 'NaN':
        restricted_stock = data_dict[item]['restricted_stock']
    if data_dict[item]['exercised_stock_options'] != 'NaN':
        exercised_stock = data_dict[item]['exercised_stock_options']
    if data_dict[item]['restricted_stock_deferred'] != 'NaN':
        restricted_deferred_stock = data_dict[item]['restricted_stock_deferred']
    data_dict[item]['total_stock_value'] = restricted_stock + exercised_stock + restricted_deferred_stock


add_feature(data_dict,'to_poi_ratio')
for item in data_dict:
    to_message = 0
    from_poi_to_this_person = 0
    if data_dict[item]['from_poi_to_this_person'] != 'NaN':
        from_poi_to_this_person = data_dict[item]['from_poi_to_this_person']
    if data_dict[item]['to_messages'] != 'NaN':
        to_message = data_dict[item]['to_messages']
    else:
        to_message = 1
    data_dict[item]['to_poi_ratio'] = from_poi_to_this_person /to_message 
    
add_feature(data_dict,'from_poi_ratio')
for item in data_dict:
    from_message = 0
    from_this_person_to_poi = 0
    if data_dict[item]['from_this_person_to_poi'] != 'NaN':
        from_this_person_to_poi = data_dict[item]['from_this_person_to_poi']
    if data_dict[item]['from_messages'] != 'NaN':
        from_message = data_dict[item]['from_messages']
    else:
        from_message = 1
    data_dict[item]['to_poi_ratio'] = from_this_person_to_poi /from_message 


### Store to my_dataset for easy export below.
my_dataset = data_dict

# Here I add the all interesting features 
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'exercised_stock_options', \
                 'bonus', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', \
                 'expenses', 'loan_advances', 'other', 'director_fees', 'deferred_income', \
                 'long_term_incentive', 'total_stock_value', 'to_poi_ratio', 'from_poi_ratio'] # You will need to use more features


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


## Feature Engineeering
from sklearn.feature_selection import SelectKBest
clf = SelectKBest(k=11).fit(features, labels)
feature_weights = {}

for idx, feature in enumerate(clf.scores_):
    feature_weights[features_list[1:][idx]] = feature
best_features = sorted(feature_weights.items(), key = lambda k: k[1], reverse = True)[:11]
new_features = []
for k, v in best_features:
    new_features.append(k)
new_features.insert(0, 'poi')

features_list = new_features

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.metrics import *

def evaluateClf(grid_search, features, labels, params, iters=20):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)] 
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]

    print "accuracy: {}".format(average(acc))
    print "precision: {}".format(average(pre))
    print "recall:    {}".format(average(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))  

# Provided to give you a starting point. Try a variety of classifiers.
''''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

param = {}
grid_search = GridSearchCV(clf, param)

print("Naive bayes model: ")
t0 = time()
evaluateClf(grid_search, features, labels, param)
print("done in %0.3fs" % (time() - t0))
'''

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
param = {'criterion': ['gini','entropy'],'min_samples_split': [2,3,4,5,6,7]}
    
grid_search = GridSearchCV(clf, param)

print("DecisionTreeClassifier model: ")
t0 = time()
evaluateClf(grid_search, features, labels, param)
print("done in %0.3fs" % (time() - t0))

'''
# Regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

clf = Pipeline(steps=[
        ('scaler', preprocessing.StandardScaler()),
        ('classifier', LogisticRegression())])

param = {'classifier__tol': [1, 0.1, 0.01, 0.001, 0.0001], 
         'classifier__C': [0.1, 0.01, 0.001, 0.0001]}

grid_search = GridSearchCV(clf, param)

print("Regression model: ")
t0 = time()
evaluateClf(grid_search, features, labels, param)
print("done in %0.3fs" % (time() - t0))

'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1, random_state=42)
    
    

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


