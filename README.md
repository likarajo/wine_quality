# Wine Quality  

Predict the quality of wine (red wine) based on different attributes.

## Goal

* Use **Random Forest** algorithm to create machine learning models
* Evaluate the model using **cross-validation**
* Select the best model using **grid-search**
* Predict using the best model

## Background  

A typical machine learning process involves **training** different models on the dataset, **evaluating** the performance of algorithm and **selecting** the one with best performance.  

There are several factors that determine which algorithm performs the best:

* Performance of the algorithm on **cross validation** set
* Choice of **hyperparameters** for the algorithm

### Cross Validation for model accuracy

The training set is used to train the model and the test set is used to evaluate the performance of the model. However, this may lead to **variance problem** where *the accuracy obtained on one test set is very different to accuracy obtained on another test set using the same algorithm*.

The solution to this is the process of **K-Fold Cross-Validation**:

* Divide the data into K folds.  
* Out of the K folds, K-1 sets are used for training while the remaining set is used for testing.  
* The algorithm is trained and tested K times, each time a new set is used as testing set while remaining sets are used for training.  
* Finally, the result of the K-Fold Cross-Validation is the average of the results obtained on each set. 

### Grid Search for Hyperparameter selection

Randomly selecting the hyperparameters for the algorithm can be exhaustive. It is also not easy to compare performance of different algorithms by randomly setting the hyper parameters because one algorithm may perform better than the other with different set of parameters. And if the parameters are changed, the algorithm may perform worse than the other algorithms.

**Grid Search** is an algorithm which automatically finds the best parameters for a particular model.

* Create a dictionary of all the hyperparameters and their corresponding set of values that are set to test for best performance.
  * The name of the dictionary items corresponds to the parameter name and the value corresponds to the list of values for the parameter.
* Create an instance of the algorithm class.
* Pass the values for the hyperparameter from the dictionary.
* Check the hyperparameters that return the highest accuracy.
* Find the accuracy obtained using the best parameters.

---

## Dependencies

* Pandas
* Scikit-learn

`pip install -r requirements.txt`

## Dataset

UCI archive ML data: https://archive.ics.uci.edu/ml/datasets/wine+quality<br>
Saved in: *data/winequality-red.csv*

## Data Preprocessing

* Separate the features and labels
* Prepare data for cross-validation
  * All the data is kept in the training set
* Scale the training data

## Implementing the algorithm

* **Random Forest Classifier**
* Estimators = 300

## Implementing Cross Validation

* Cross Validation **Accuracy**
* Number of **Folds** = 5

## Parameter selection for best model

* **Grid Search**
* Estimators: 100, 300, 500, 700, 1000
* Criteria: gini, entropy
* With and without bootstrap

## Conclusion

* **K-Fold Cross-Validation** is used to *evaluate performance of a model* by handling the ***variance problem*** of the result set.
* To *identify the best algorithm and best parameters*, the **Grid Search** algorithm is used.
