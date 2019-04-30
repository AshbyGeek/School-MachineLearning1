#%% [markdown]
#  Project Statement
#  ============
#  As with Project 1, apply the ideas of ch. 1 - 3 as appropriate.
#  Develop and demonstrate your capabilities with:
#   * Regression (ch. 4)
#   * Support Vector Machines (SVM's) (ch. 5)
# 
# 
#   Starting Point
#   --------------
#   To start this project I'm going to pull from the Titanic dataset part of the previous, since I was a little disappointed that Derek Byrnes got a higher score than me. Friendly rivalry and what not. :) So now that we're working together, I'll see if I can pull some of his techniques in to produce a better score.

#%%
import pandas

import os
os.chdir(r"Z:\DannyProgramming\Mercer\machineLearning1")

raw_training = pandas.read_csv("titanic/train.csv")
raw_test = pandas.read_csv("titanic/test.csv")
example_output = pandas.read_csv("titanic/gender_submission.csv")

y = raw_training["Survived"].copy()
X = raw_training.drop("Survived", axis=1)
y.head()


#%%
X.head()

#%% [markdown]
# A transform for using 

#%%
from sklearn.base import BaseEstimator, TransformerMixin
import re
import pandas.api.types as ptypes

class RegexTransform(TransformerMixin):
    '''
    Named groups within the regular expression are taken to be values to be put into
    a bag of words and one-hot encoded.
    '''
    def __init__(self, regex=".*", keepOriginalCols=True, columns='All'):
        '''

        '''
        self.pattern = re.compile(regex)
        self.keepOriginalCols = keepOriginalCols
        self.columns = columns
        
    def fit(self, X, y=None):
        self._fitVals = set()
        if (isinstance(X, pandas.DataFrame)):
            tmpCols = self.columns
            if self.columns == 'All':
                tmpCols = X.columns
            tmpdf = X[tmpCols]

            for col in tmpdf:
                self._fitVals |= self._findValsInSeries(X[col])
        self._newCols = list(self._fitVals)
        self._newCols.sort()
        return self
    
    def transform(self, X, y=None):
        if (isinstance(X, pandas.DataFrame)):
            newCols = pandas.DataFrame(data=0, index=X.index, columns=self._newCols, dtype="int")
            
            tmpCols = self.columns
            if self.columns == 'All':
                tmpCols = X.columns
            tmpdf = X[tmpCols]

            for index,row in tmpdf.iterrows():
                vals = self._findValsInSeries(row)
                for val in vals:
                    newCols.at[index,val] = 1

            newX = pandas.concat([X.copy(),newCols], axis=1)
            if (not self.keepOriginalCols):
                newX = newX.drop(tmpCols, axis=1)
            return newX

    def _findValsInSeries(self, series):
        vals = set()
        for val in series:
            matches = self.pattern.search(val)
            for value in matches.groupdict().values():
                if not value is None:
                    vals.add(value)
        return vals

    def get_feature_names(self, X):
        features = []
        if isinstance(X, pandas.DataFrame):
            if (self.keepOriginalCols or self.columns != 'All'):
                for feature in X:
                    if self.columns != 'All' and not feature in self.columns:
                        features.append(X.get_feature_names)
        elif isinstance(X, pandas.Series):
            if (self.keepOriginalCols):
                features.append(X.name)
        features.append(self._newCols)
        return features
            
regexTest = RegexTransform(regex=r"(?P<surname>.*?),[^\(]*(?:\(.* (?P<maiden>.*)\))?", keepOriginalCols=True, columns=["Name"])
results = regexTest.fit_transform(X)
print(results.head())


#%%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer

def CreateTitanicPipeline():
        numeric_steps=[
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('discretizer', KBinsDiscretizer())
        ]
        numeric_pipeline = Pipeline(numeric_steps)

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown ='ignore'))
        ])

        namesPipeline = Pipeline(steps=[
            ('lastName', RegexTransform(regex=r"(?P<surname>.*?),[^\(]*(?:\(.* (?P<maiden>.*)\))?", keepOriginalCols=False))
        ])

        return ColumnTransformer(sparse_threshold=0,transformers=[
            ('numerical', numeric_pipeline, ["Age", "Fare"]),
            ('categorical', categorical_pipeline, ["Sex", "Embarked", "Pclass"]),
            ('passthrough', "passthrough", ["SibSp", "Parch"]),
            ('names', namesPipeline, ["Name"])
        ])

#titanic_pipeline = CreateTitanicPipeline()
#for key in titanic_pipeline.get_params():
    #print(key)

#%% [markdown]
# Allright, now to see if things have gotten better!

#%%
def CreateTitanicPipeline2():
    numeric_steps=[
        ('imputer', SimpleImputer()),
        ('discretizer', KBinsDiscretizer()),
    ]

    categorical_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown ='ignore')),
    ]

    names_steps=[
        ('lastName', RegexTransform(regex=r"(?P<surname>.*?),[^\(]*(?:\(.* (?P<maiden>.*)\))?", keepOriginalCols=False, columns=["Name"]))
    ]

    return ColumnTransformer(sparse_threshold=0,transformers=[
        ('numericAge', Pipeline(numeric_steps), ["Age"]),
        ('numericFare', Pipeline(numeric_steps), ["Fare"]),
        ('categorical', Pipeline(categorical_steps), ["Sex", "Embarked", "Pclass"]),
        ('passthrough', "passthrough", ["SibSp", "Parch"]),
        ('names', Pipeline(names_steps), ["Name"]),
    ])

for key in CreateTitanicPipeline2().get_params():
    print(key)


#%%
import pprint
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
def tune_decisionTree2(data, labels):
    param_grid = { 
        "transform__numericAge__discretizer__n_bins": [3, 5, 8, 10],
        "transform__numericAge__discretizer__strategy": ["kmeans", "quantile"],
        "transform__numericAge__discretizer__encode":["onehot", "ordinal"],
        "transform__numericFare__discretizer__n_bins": [2,4,6,8],
        "transform__numericFare__discretizer__strategy": ["kmeans", "quantile"],
        "transform__numericFare__discretizer__encode":["onehot", "ordinal"],
        "classifier__splitter": ["random", "best"],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__min_samples_split": [8, 10, 12],
        "classifier__min_samples_leaf": [1,3,5],
        "classifier__random_state":[42]
    }

    pipeline = Pipeline([
        ("transform", CreateTitanicPipeline2()),
        ("classifier", DecisionTreeClassifier())
    ])
    search = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=3, iid=False)
    search.fit(data, labels)
    pp = pprint.PrettyPrinter(indent=4)
    print("Best Params: ")
    pp.pprint(search.best_params_)
    print("Best Score: ", search.best_score_)
    print("Refit Time: ", search.refit_time_)
    return (search.best_estimator_, search)

best_dt2, search2 = tune_decisionTree2(X, y)


