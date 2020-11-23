# IBM-Machine-Learning

## Summary/Review

### Retrieving Data
You can retrieve data from multiple sources:

SQL databases
NoSQL databases
APIs
Cloud data sources
The two most common formats for delimited data flat files are comma separated (csv) and tab separated (tsv). It is also possible to use special characters as separators.

SQL represents a set of relational databases with fixed schemas.

Reading in Database Files
The steps to read in a database file using the sqlite library are:

create a path variable that references the path to your database
create a connection variable that references the connection to your database
create a query variable that contains the SQL query that reads in the data table from your database
create an observations variable to assign the read_sql functions from pandas package
create a tables variable to read in the data from the table sqlite_master
JSON files are a standard way to store data across platforms. Their structure is similar to Python dictionaries.

NoSQL databases are not relational and vary more in structure. Most NoSQL databases store data in JSON format.

Data Cleaning
Data Cleaning is important because messy data will lead to unreliable outcomes.Some common issues that make data messy are: duplicate or unnecessary data, inconsistent data and typos, missing data, outliers, and data source issues.

You can identify duplicate or unnecessary dataCommon policies to deal with missing data are:remove a row with missing columns, impute the missing data, and mask the data by creating a category for missing values.

Common methods to find outliers are: through plots, statistics, or residuals.

Common policies to deal with outliers are: remove outliers, impute them, use a variable transformation, or use a model that is resistant to outliers.

Exploratory Data Analysis
EDA is an approach to analyzing data sets that summarizes their main characteristics, often using visual methods. It helps you determine if the data is usable as-is, or if it needs further data cleaning.

EDA is also important in the process of identifying patterns, observing trends, and formulating hypothesis.

Common summary statistics for EDA include finding summary statistics and producing visualizations.

Feature Engineering and Variable Transformation
Transforming variables helps to meet the assumptions of statistical models. A concrete example is a linear regression, in which you may transform a predictor variable such that it has a linear relation with a target variable.

Common variable transformations are: calculating log transformations and polynomial features, encoding a categorical variable, and scaling a variable.

### Estimation and Inference
Inferential Statistics consist in learning characteristics of the population from a sample. The population characteristics are parameters, while the sample characteristics are statistics. A parametric model, uses a certain number of parameters like mean and standard deviation.

The most common way of estimating parameters in a parametric model is through maximum likelihood estimation.

Through a hypothesis test, you test for a specific value of the parameter.

Estimation represents a process of determining a population parameter based on a model fitted to the data.

The most common distribution functions are: uniform, normal, log normal, exponential, and poisson.

A frequentist approach focuses in observing man repeats of an experiment. A bayesian approach describes parameters through probability distributions.

Hypothesis Testing
A hypothesis is a statement about a population parameter. You commonly have two hypothesis: the null hypothesis and the alternative hypothesis.

A hypothesis test gives you a rule to decide for which values of the test statistic you accept the null hypothesis and for which values you reject the null hypothesis and accept he alternative hypothesis.

A type 1 error occurs when an effect is due to chance, but we find it to be significant in the model.

A type 2 error occurs when we ascribe the effect to chance, but the effect is non-coincidental.

Significance level and p-values
A significance level is a probability threshold below which the null hypothesis can be rejected. You must choose the significance level before computing the test statistic. It is usually .01 or .05.

A p-value is the smallest significance level at which the null hypothesis would be rejected. The confidence interval contains the values of the statistic for which we accept the null hypothesis.

Correlations are useful as effects can help predict an outcome, but correlation does not imply causation.

When making recommendations, one should take into consideration confounding variables and the fact that correlation across two variables do not imply that an increase or decrease in one of them will drive an increase or decrease of the other.

Spurious correlations happen in data. They are just coincidences given a particular data sample.

### Introduction to Supervised Machine Learning
The types of supervised Machine Learning are:

Regression, in which the target variable is continuous
Classification, in which the target variable is categorical
To build a classification model you need:

Features that can be quantified
A labeled target or outcome variable
Method to measure similarity 
Linear Regression
A linear regression models the relationship between a continuous variable and one or more scaled variables.It is usually represented as a dependent function equal to the sum of a coefficient plus scaling factors times the independent variables. 

Residuals are defined as the difference between an actual value and a predicted value. 

A modeling best practice for linear regression is:

Use cost function to fit the linear regression model
Develop multiple models
Compare the results and choose the one that fits your data and whether you are using your model for prediction or interpretation. 
Three common measures of error for linear regressions are:

Sum of squared Error (SSE)
Total Sum of Squares (TSS)
Coefficient of Determination (R2)

### Training and Test Splits
Splitting your data into a training and a test set can help you choose a model that has better chances at generalizing and is not overfitted.

The training data is used to fit the model, while the test data is used to measure error and performance. 

Training error tends to decrease with a more complex model.Cross validation error generally has a u-shape. It decreases with more complex models, up to a point in which it starts to increase again. 

Cross Validation
The three most common cross validation approaches are:

k-fold cross validation
leave one out cross validation
stratified cross validation
Polynomial Regression
Polynomial terms help you capture nonlinear effects of your features. 

### Regularization Techniques
Three sources of error for your model are: bias, variance, and, irreducible error.

Regularization is a way to achieve building simple models with relatively low error. It helps you avoid overfitting by penalizing high-valued coefficients. It reduces parameters and shrinks the model.

Regularization adds an adjustable regularization strength parameter directly into the cost function.

Regularization performs feature selection by shrinking the contribution of features, which can prevent overfitting.

In Ridge Regression, the complexity penalty λ is applied proportionally to squared coefficient values.

–  The penalty term has the effect of “shrinking” coefficients toward 0.

–  This imposes bias on the model, but also reduces variance.

–  We can select the best regularization strength lambda via cross-validation.

–  It’s a best practice to scale features (i.e. using StandardScaler) so penalties aren’t impacted by variable scale.

In LASSO regression: the complexity penalty λ (lambda) is proportional to the absolute value of coefficients. LASSO stands for : Least Absolute Shrinkage and Selection Operator.

–  Similar effect to Ridge in terms of complexity tradeoff: increasing lambda raises bias but lowers variance.

–  LASSO is more likely than Ridge to perform feature selection, in that for a fixed λ, LASSO is more likely to result in coefficients being set to zero.

Elastic Net combines penalties from both Ridge and LASSO regression. It requires tuning of an additional parameter that determines emphasis  of L1 vs. L2 regularization penalties.

LASSO’s feature selection property yields an interpretability advantage, but may underperform if the target truly depends on many of the features.

Elastic Net, an alternative hybrid approach, introduces a new parameter α (alpha) that determines a weighted average of L1 and L2 penalties.

Regularization techniques have an analytical, a geometric, and a probabilistic interpretation.


### Classification Problems
The two main types of supervised learning models are:

Regression models, which predict a continuous outcome
Classification models, which predict a categorical outcome.
The most common models used in supervised learning are:

Logistic Regression
K-Nearest Neighbors
Support Vector Machines
Decision Tree
Neural Networks
Random Forests
Boosting
Ensemble Models
With the exception of logistic regression, these models are commonly used for both regression and classification. Logistic regression is most common for dichotomous and nominal dependent variables.

Logistic Regression
Logistic regression is a type of regression that models the probability of a certain class occurring given other independent variables.It uses a logistic or logit function to model a dependent variable. It is a very common predictive model because of its high interpretability.

Classification Error Metrics
A confusion matrix tabulates true positives, false negatives, false positives and true negatives. Remember that the false positive rate is also known as a type I error. The false negatives are also known as a type II error.

Accuracy is defined as the ratio of true postives and true negatives divided by the total number of observations. It is a measure related to predicting correctly positive and negative instances.

Recall or sensitivity identifies the ratio of true positives divided by the total number of actual positives. It quantifies the percentage of positive instances correctly identified.

Precision is the ratio of true positive divided by total of predicted positives. The closer this value is to 1.0, the better job this model does at identifying only positive instances.

Specificity is the ratio of true negatives divided by the total number of actual negatives. The closer this value is to 1.0, the better job this model does at avoiding false alarms.

The receiver operating characteristic (ROC) plots the true positive rate (sensitivity) of a model vs. its false positive rate (1-sensitivity).

The area under the curve of a ROC plot is a very common method of selecting a classification methods. The precision-recall curve measures the trade-off between precision and recall.

The ROC curve generally works better for data with balanced classes, while the precision-recall curve generally works better for data with unbalanced classes.

### K Nearest Neighbor Methods for Classification
K nearest neighbor methods are useful for classification. The elbow method is frequently used to identify a model with low K and low error rate.

These methods are popular due to their easy computation and interpretability, although it might take time scoring new observations, it lacks estimators, and might not be suited for large data sets.

### SVM
The main idea behind support vector machines is to find a hyperplane that separates classes by determining decision boundaries that maximize the distance between classes.

When comparing logistic regression and SVMs, one of the main differences is that the cost function for logistic regression has a cost function that decreases to zero, but rarely reaches zero. SVMs use the Hinge Loss function as a cost function to penalize misclassification. This tends to lead to better accuracy at the cost of having less sensitivity on the predicted probabilities.

Regularization can help SVMs generalize better with future data.

By using gaussian kernels, you transform your data space vectors into a different coordinate system, and may have better chances of finding a hyperplane that classifies well your data.SVMs with RBFs Kernels are slow to train with data sets that are large or have many features.  

### Decision Tree

Decision trees split your data using impurity measures. They are a greedy algorithm and are not based on statistical assumptions.
The most common splitting impurity measures are Entropy and Gini index.Decision trees tend to overfit and to be very sensitive to different data.
Cross validation and pruning sometimes help with some of this.
Great advantages of decision trees are that they are really easy to interpret and require no data preprocessing.  

#### Classification Error vs Entropy
https://sebastianraschka.com/faq/docs/decisiontree-error-vs-entropy.html

### Ensemble Based Methods and Bagging
Tree ensembles have been found to generalize well when scoring new data. Some useful and popular tree ensembles are bagging, boosting, and random forests. Bagging, which combines decision trees by using bootstrap aggregated samples. An advantage specific to bagging is that this method can be multithreaded or computed in parallel. Most of these ensembles are assessed using out-of-bag error.

Random Forest
Random forest is a tree ensemble that has a similar approach to bagging. Their main characteristic is that they add randomness by only using a subset of features to train each split of the trees it trains. Extra Random Trees is an implementation that adds randomness by creating splits at random, instead of using a greedy search to find split variables and split points.

Boosting
Boosting methods are additive in the sense that they sequentially retrain decision trees using the observations with the highest residuals on the previous tree. To do so, observations with a high residual are assigned a higher weight.

Gradient Boosting
The main loss functions for boosting algorithms are:

0-1 loss function, which ignores observations that were correctly classified. The shape of this loss function makes it difficult to optimize.
Adaptive boosting loss function, which has an exponential nature. The shape of this function is more sensitive to outliers.
Gradient boosting loss function. The most common gradient boosting implementation uses a binomial log-likelihood loss function called deviance. It tends to be more robust to outliers than AdaBoost.
The additive nature of gradient boosting makes it prone to overfitting. This can be addressed using cross validation or fine tuning the number of boosting iterations. Other hyperparameters to fine tune are:

learning rate (shrinkage)
subsample
number of features.
Stacking
Stacking is an ensemble method that combines any type of model by combining the predicted probabilities of classes. In that sense, it is a generalized case of bagging. The two most common ways to combine the predicted probabilities in stacking are: using a majority vote or using weights for each predicted probability.

### Modeling Unbalanced Classes
Classification algorithms are built to optimize accuracy, which makes it challenging to create a model when there is not a balance across the number of observations of different classes. Common methods to approach balancing the classes are:

 * Downsampling or removing observations from the most common class
Upsampling or duplicating observations from the rarest class or classes
A mix of downsampling and upsampling
Modeling Approaches for Unbalanced Classes
Specific algorithms to upsample and downsample are:

Stratified sampling
Random oversampling
Synthetic oversampling, the main two approaches being Synthetic Minority Oversampling Technique (SMOTE) and Adaptive Synthetic sampling (ADASYN)
Cluster Centroids implementations like NearMiss, Tomek Links, and Nearest Neighbors  

#### Unbalanced data
https://www.svds.com/learning-imbalanced-classes/

