# Imbalanced Data  <br>
## An overview of working with imbalanced data<br>
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
The number of our observations in our oversampled data is dependent upon the sampling strategy parameter. The number of non events stays constant and the number of events is equal to *(NumberofNonEvents-NumberofEvents) X SamplingStrategyParameter* for a total number of observations of *(Number of Non-Events-Number of Events) X sampling strategy parameter + Number of Non-Events*. In this case, we have 5,700 events and 5,6250 non-events.<br>
The number of observations in our undersampled data is 11,400. This is derived from the equation *NumberofEvents/SamplingStrategyParameter*. 
```
#Over Sampled Data
oversample = RandomOverSampler(sampling_strategy=.5,  random_state=2)
X_over, y_over = oversample.fit_resample(X_train, y_train)
#Under Sampled Data
undersample = RandomUnderSampler(sampling_strategy=1,  random_state=2)
X_under, y_under = undersample.fit_resample(X_train, y_train)
```
We will create 3 Logit Models with our 3 data sets, and determine which one performs best.<br>
## Logit Model with original distribution
```
LogitModel=LogisticRegression()
LogitModel=LogisticRegression()
LogitModel.fit(X_train,y_train)
y_pred=LogitModel.predict(X_test)
accuracy_score(y_test,y_pred)
NormalLogitAccuracy=accuracy_score(y_test,y_pred)
NormalLogitClass=classification_report(y_test,y_pred)
NormalLogitMx=confusion_matrix(y_test,y_pred).ravel()
NormalLogitMCC=matthews_corrcoef(y_test,y_pred)
print(i for i in ['NORMAL LOGIT',NormalLogitAccuracy,NormalLogitClass,NormalLogitMx,NormalLogitMCC])
```
| Contingency Table|Accuracy| Precision| Recall|F1|Matthews Coef.|
|------------------|--------|----------|-------|--|--------------|
|<table> <thead> <tr>  <th></th> <th>Predicted Events</th>    <th>Predicted Non-Events</th>    <tbody>  <tr>  <td>Actual Events</td>   <td>34</td> <td>1,765</td> </tr>  <tr> <td>Actual Non-Events</td>  <td>2</td> <td>16,549</td> </tbody> </table>   | 0.905 |  0.94      | 0.2      | 0.04 | 0.126             |
  
If one is just looking at just the accuracy, they would think this a good model. However, we know that only 10% of the observations are events. If we just predicted that all observations would be non-events, we would get the same accuracy, so that is not a good metric to use when you have sparse data.<br>
Our Precision is 0.94, this looks good, but look at the contingency table. The model only predicted an event on 36 observations. <br>
The recall is .02. Of 1,799 events in the sample we only predicted 34 of them.<br>
The F1 Score is .04 which is low and the Matthews Correlation Coefficient is 0.12 which is fairly low.<br>
  
## Logit Model with Over Sampled data
```
LogitModelOver=LogisticRegression()
LogitModelOver.fit(X_over,y_over)
```
We use the estimators that were derived from the over sampled data on our original test data set and see how it performs. 
```
y_pred_over=LogitModelOver.predict(X_test)
OverLogitAccuracy=accuracy_score(y_test,y_pred_over)
OverLogitClass=classification_report(y_test,y_pred_over)
OverLogitMx=confusion_matrix(y_test,y_pred_over).ravel()
OverLogitMCC=matthews_corrcoef(y_test,y_pred_over)
[print(i) for i in ['OVER SAMPLE',OverLogitAccuracy,OverLogitClass,OverLogitMx,OverLogitMCC]]
```
| Contingency Table|Accuracy| Precision| Recall|F1|Matthews Coef.|
|------------------|--------|----------|-------|--|--------------|
|<table> <thead> <tr>  <th></th> <th>Predicted Events</th>    <th>Predicted Non-Events</th>    <tbody>  <tr>  <td>Actual Events</td>   <td>336</td> <td>1,463</td> </tr>  <tr> <td>Actual Non-Events</td>  <td>869</td> <td>16,082</td> </tbody> </table>   | 0.87 |  0.28      | 0.19      | 0.22 | 0.16             |
  
Our Accuracy is 0.87, which is a decrease, but the model was more aggressive in predicting events. This lowered out precision and raised our recall.<br>
Our Precision is 0.28.  The model  predicted an event on 1,205 observations and was correct 336 times. <br>
The recall is .19. Of 1,799 events in the sample we only predicted 336 of them.<br>
The F1 Score is .22 which is low and the Matthews Correlation Coefficient is 0.16 which is fairly low. However, this is still an improvement over our first model<br>
  
## Logit Model with Under Sampled data
 ```
LogitModelUnder=LogisticRegression()
LogitModelUnder.fit(X_under,y_under)
 ```
 We use the estimators that were derived from the under sampled data on our original test data set and see how it performs. 
