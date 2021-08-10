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
X,y = make_classification(n_samples=2500,n_informative=10 ,n_features=40, n_redundant=20,n_clusters_per_class=2, weights=[0.9], flip_y=0,class_sep=.5,random_state=15)
X_train,X_test, y_train, y_test = train_test_split(X, y,  random_state=2)
```
The number of our observations in our oversampled data is dependent upon the sampling strategy parameter. The number of non events stays constant and the number of events is equal to *(NumberofNonEvents-NumberofEvents) X SamplingStrategyParameter* for a total number of observations of *(Number of Non-Events-Number of Events) X sampling strategy parameter + Number of Non-Events*. In this case, we have 845 events and 1,691 non-events.<br>
The number of observations in our undersampled data is 368. This is derived from the equation *NumberofEvents/SamplingStrategyParameter*. 
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
|<table> <thead> <tr>  <th></th> <th>Predicted Events</th>    <th>Predicted Non-Events</th>    <tbody>  <tr>  <td>Actual Events</td>   <td>7</td> <td>58</td> </tr>  <tr> <td>Actual Non-Events</td>  <td>4</td> <td>556</td> </tbody> </table>   | 0.90 |  0.64      | 0.11      | 0.18 | 0.23             |
  
If one is just looking at just the accuracy, they would think this a good model. However, we know that only 10% of the observations are events. If we just predicted that all observations would be non-events, we would get the same accuracy, so that is not a good metric to use when you have sparse data.<br>
Our Precision is 0.94, this looks good, but look at the contingency table. The model only predicted an event on 11 observations. <br>
The recall is .11. Of 65 events in the sample we only predicted 7 of them.<br>
The F1 Score is .18 and the Matthews Correlation Coefficient is 0.23 w.<br>
  
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
|<table> <thead> <tr>  <th></th> <th>Predicted Events</th>    <th>Predicted Non-Events</th>    <tbody>  <tr>  <td>Actual Events</td>   <td>34</td> <td>31</td> </tr>  <tr> <td>Actual Non-Events</td>  <td>61</td> <td>499</td> </tbody> </table>   | 0.85 |  0.36      | 0.52      | 0.42 | 0.35             |
  
Our Accuracy is 0.85, which is a decrease<br>
Our Precision is 0.28.  The model  predicted an event on 95 observations and was correct 34 times. <br>
The recall is .19. Of 65 events in the sample we only predicted 34 of them.<br>
The F1 Score is .42 and the Matthews Correlation Coefficient is 0.35 . This is still an improvement over our first model<br>
  
## Logit Model with Under Sampled data
 ```
LogitModelUnder=LogisticRegression()
LogitModelUnder.fit(X_under,y_under)
 ```
 We use the estimators that were derived from the under sampled data on our original test data set and see how it performs. 
```
y_pred_under=LogitModelUnder.predict(X_test)
UnderLogitAccuracy=accuracy_score(y_test,y_pred_under)
UnderLogitClass=classification_report(y_test,y_pred_under)
UnderLogitMx=confusion_matrix(y_test,y_pred_under).ravel()
UnderLogitMCC=matthews_corrcoef(y_test,y_pred_under)
[print(i) for i in ['UNDER SAMPLE',UnderLogitAccuracy,UnderLogitClass,UnderLogitMx,UnderLogitMCC]]
 ```
| Contingency Table|Accuracy| Precision| Recall|F1|Matthews Coef.|
|------------------|--------|----------|-------|--|--------------|
|<table> <thead> <tr>  <th></th> <th>Predicted Events</th>    <th>Predicted Non-Events</th>    <tbody>  <tr>  <td>Actual Events</td>   <td>51</td> <td>14</td> </tr>  <tr> <td>Actual Non-Events</td>  <td>147</td> <td>413</td> </tbody> </table>   | 0.74 |  0.26      | 0.78      | 0.39 | 0.34             |

Our Accuracy is 0.74, which is a lower than the previous two models, but the model was more aggressive in predicting events. This lowered out precision and raised our recall.<br>
Our Precision is 0.26.  The model  predicted an event on 147 observations and was correct 51 times. <br>
The recall is .78. Of 65 events in the sample we only predicted 51 of them.<br>
The F1 Score is .39 and the Matthews Correlation Coefficient is 0.34 . This is an improvement over our first model, but slightly worse performing than our second model.<br>
   
## What Model to Use?
There is no right answer. It deponds on the situtation. If the cost of taking action on a predicted event is low, a larger recall is better. If the cost of taking action is higher, a higher precision is better. You should also look at the contingency tables as well, you could have a precision of 1.0, but if it's only making predicitions on 1 observation, the model not very useful.
 
