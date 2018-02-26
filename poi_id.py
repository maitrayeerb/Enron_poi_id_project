#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import tester
from tester import dump_classifier_and_data
import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
                 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
                 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_df = pd.DataFrame.from_dict(data_dict, orient='index')
print ("data_df_raw")
print (data_df.info())


data_df = data_df[features_list]
data_df = data_df.replace('NaN', 0)

# outlier calculation:
print ("outliers")
outliers = data_df.quantile(0.75) + 1.5*(data_df.quantile(0.75)-data_df.quantile(0.25))

data_Outliers =  pd.DataFrame((data_df[1:] > outliers[1:]).sum(axis = 1), columns = ['# of outliers'])
data_df_outliers = pd.concat([data_df, data_Outliers], axis=1)
print ("Outliers")
print (data_Outliers.sort_values('# of outliers',  ascending = [0]).head(10))

data_df = data_df.drop(["TOTAL"])
print ("Removed outlier TOTAL")


## ## Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

data_df.loc[:,('fraction_from_this_person_to_poi')] = 0
data_df['fraction_from_this_person_to_poi'].loc[data_df['from_messages']!=0] \
    = data_df['from_this_person_to_poi']/data_df['from_messages']

data_df.loc[:,('fraction_from_poi_to_this_person')] = 0
data_df['fraction_from_poi_to_this_person'].loc[data_df['to_messages']!=0] \
    = data_df['from_poi_to_this_person']/data_df['to_messages']

features_list.append("fraction_from_poi_to_this_person")
features_list.append("fraction_from_this_person_to_poi")

print ("Feature List:")
print (features_list)

### Extract features and labels from dataset for local testing

features_list_test = features_list
my_dataset_test = data_df
my_dataset_test = my_dataset_test.to_dict(orient = 'index')
data = featureFormat(my_dataset_test, features_list_test, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

clf_list = []
from sklearn.naive_bayes import GaussianNB
clf_list.append(GaussianNB())

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf_list.append(RandomForestClassifier(n_estimators=10, min_samples_split=10))
clf_list.append(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200))

for clf in clf_list:
    dump_classifier_and_data(clf, my_dataset_test, features_list_test)
    tester.main()


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = GaussianNB()

# data set standartization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import imputation

scaler = StandardScaler()

data_df = data_df.replace('NaN', 0)
data_df_norm = data_df[features_list[1:]]
imp = imputation.Imputer(missing_values='NaN', strategy='median', axis=0)
data_df_norm = imp.fit_transform(data_df_norm) #.ix[:,1:]
data_df_norm = scaler.fit_transform(data_df_norm)

# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

k_value = 10
selector = SelectKBest(f_classif, k=k_value)
data_df_select = pd.DataFrame(selector.fit_transform(data_df_norm, data_df.poi),
                                index = data_df.index)
features_list_selectk = features_list[1:]

mask = selector.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, features_list_selectk):
    if bool:
        new_features.append(feature)

print ("Selected Features:")
print (new_features)

my_dataset_df = data_df_select
my_dataset_df.columns = new_features
features_list_select = ['poi'] + new_features
my_dataset_df.insert(0, 'poi', data_df.poi)

#print ("my_dataset_df:")
#print (my_dataset_df.head(1))

# pipeline and gridsearch PCA with Gaussian NB:
print("Gaissian NB wtih PCA:")
my_dataset_pca_gn = my_dataset_df.to_dict(orient = 'index')
data = featureFormat(my_dataset_pca_gn, features_list_select, sort_keys = True)
labels, features = targetFeatureSplit(data)
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('gassuian', GaussianNB())])
n_components = [2, 3, 4, 5, 6, 7, 8,9,10]
param = dict(pca__n_components=n_components)
estimator = GridSearchCV(pipe,param_grid=param)
estimator.fit(features, labels)
clf = estimator.best_estimator_
print ("Best PCA Gassian estimator:")
print (estimator.best_params_)


pca = PCA(n_components=6)
print ("pcs n_comp = 2")
dump_classifier_and_data(clf, my_dataset_pca_gn, features_list_select)
tester.main()


# Adaboost:
print("Adaboost:")
my_dataset_adb = my_dataset_df # data_df pd.DataFrame(data_df_norm, index = data_df.index)

my_dataset_test = my_dataset_adb.to_dict(orient = 'index')
features_list_test = features_list_select # features_list
data = featureFormat(my_dataset_test, features_list_test, sort_keys = True)
labels, features = targetFeatureSplit(data)


clf = AdaBoostClassifier(random_state = 75)
clf.fit(features,labels)

# selecting the features with non null importance, sorting and creating features_list for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([my_dataset_adb.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

print ("features_importance")
print(features_importance)

# number of features for best result was found iteratively
print ("Number of important features chosen: 3")
features_list_test = features_list[:3]
my_dataset_test = my_dataset_adb[features_list_test].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset_test, features_list_test)
tester.main()

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)

