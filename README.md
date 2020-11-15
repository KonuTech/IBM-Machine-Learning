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
