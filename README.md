# Imbalanced Data <br>
##An overview of working with imbalanced data<br>
Import all python packages we will be using
```
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
```
We will use SciKit-Learn to create a dataframe with 40 variables, 10 of which have relevent information, 10% of the observations are classified as events, and the clusters are closer together. <br>
We will create 3 training data sets. One that will have the share the same distribution as the original data, one that uses oversampled data, and one that uses undersampled data.
```
X,y = make_classification(n_samples=75000,n_informative=10 ,n_features=40, n_redundant=20,n_clusters_per_class=2, weights=[0.9], flip_y=0,class_sep=.25,random_state=15)
X_train,X_test, y_train, y_test = train_test_split(X, y,  random_state=2)
```
The number of our observations in our oversampled data is dependent upon the sampling strategy parameter. The number of non events stays constant and the number of events is equal to $$(Number of Non-Events-Number of Events) *sampling strategy parameter* $$
```
#Over Sampled Data
oversample = RandomOverSampler(sampling_strategy=.5,  random_state=2)
X_over, y_over = oversample.fit_resample(X_train, y_train)
#Under Sampled Data
undersample = RandomUnderSampler(sampling_strategy=1,  random_state=2)
X_under, y_under = undersample.fit_resample(X_train, y_train)
```
