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

### Hypothesis Testing
A hypothesis is a statement about a population parameter. You commonly have two hypothesis: the null hypothesis and the alternative hypothesis.

A hypothesis test gives you a rule to decide for which values of the test statistic you accept the null hypothesis and for which values you reject the null hypothesis and accept he alternative hypothesis.

A type 1 error occurs when an effect is due to chance, but we find it to be significant in the model.

A type 2 error occurs when we ascribe the effect to chance, but the effect is non-coincidental.

### Significance level and p-values
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

### Random Forest
Random forest is a tree ensemble that has a similar approach to bagging. Their main characteristic is that they add randomness by only using a subset of features to train each split of the trees it trains. Extra Random Trees is an implementation that adds randomness by creating splits at random, instead of using a greedy search to find split variables and split points.

### Boosting
Boosting methods are additive in the sense that they sequentially retrain decision trees using the observations with the highest residuals on the previous tree. To do so, observations with a high residual are assigned a higher weight.

### Gradient Boosting
The main loss functions for boosting algorithms are:

* 0-1 loss function, which ignores observations that were correctly classified. The shape of this loss function makes it difficult to optimize.
* Adaptive boosting loss function, which has an exponential nature. The shape of this function is more sensitive to outliers.
* Gradient boosting loss function. The most common gradient boosting implementation uses a binomial log-likelihood loss function called deviance. It tends to be more robust to outliers than AdaBoost.
The additive nature of gradient boosting makes it prone to overfitting. This can be addressed using cross validation or fine tuning the number of boosting iterations. Other hyperparameters to fine tune are:

* learning rate (shrinkage)
* subsample
* number of features.
* Stacking
Stacking is an ensemble method that combines any type of model by combining the predicted probabilities of classes. In that sense, it is a generalized case of bagging. The two most common ways to combine the predicted probabilities in stacking are: using a majority vote or using weights for each predicted probability.

### Modeling Unbalanced Classes
Classification algorithms are built to optimize accuracy, which makes it challenging to create a model when there is not a balance across the number of observations of different classes. Common methods to approach balancing the classes are:

 * Downsampling or removing observations from the most common class
 * Upsampling or duplicating observations from the rarest class or classes
 * A mix of downsampling and upsampling
### Modeling Approaches for Unbalanced Classes
 * Specific algorithms to upsample and downsample are:

 * Stratified sampling
 * Random oversampling
 * Synthetic oversampling, the main two approaches being Synthetic Minority Oversampling Technique (SMOTE) and Adaptive Synthetic sampling (ADASYN)
 * Cluster Centroids implementations like NearMiss, Tomek Links, and Nearest Neighbors  

#### Unbalanced data
https://www.svds.com/learning-imbalanced-classes/

### Unsupervised Learning Algorithms
Unsupervised algorithms are relevant when we don’t have an outcome or labeled variable we are trying to predict.

They are helpful to find structures within our data set and when we want to partition our data set into smaller pieces.   

Types of Unsupervised Learning:

Type of Unsupervised Learning	Data	Example	Algorithms
Clustering	Use unlabeled data, Identify unknown structure in data	Segmenting costumers into different groups	K-means, Hierarchical Agglomerative Clustering, DBSCAN, Mean shift
Dimensionality Reduction	Use structural characteristics to simplify data	Reducing size without losing too much information from our original data set	Principal Components Analysis, Non-negative Matrix, Factorization
Dimensionality reduction is important in the context of large amounts of data.

#### The Curse of Dimensionality

In theory, a large number of features should improve performance. In theory, as models have more data to learn from, they should be more successful. But in practice, too many features lead to worse performance. There are several reasons why too many features end up leading to worse performance. If you have too many features, several things can be wrong, for example: 

-        Some features can be spurious correlations, which means they correlate into the data set but not outside your data set as long as new data comes in. 

-        Too many features create more noise than signal.

-        Algorithms find hard to sort through non meaningful features if you have too many features. 

-        The number of training examples required increases exponentially with dimensionality.

-        Higher dimensions slows performance.

-        Larger data sets are computationally more expensive.

-        Higher incidence of outliers. 

To fix these problems in real life, it's best to reduce the dimension of the data set. 

Similar to feature selection, you can use Unsupervised Machine Learning models such as Principal Components Analysis.

#### Common uses of clustering cases in the real world
1.     Anomaly detection

Example: Fraudulent transactions.

Suspicious fraud patterns such as small clusters of credit card transactions with high volume of attempts, small amounts, at new merchants. This creates a new cluster and this is presented as an anomaly so perhaps there’s fraudulent transactions happening. 

2.     Customer segmentation

You could segment the customers by recency, frequency, average amount of visits in the last 3 months. Or another common type of segmentation is by demographics and the level of engagement, for example: single costumers, new parents, empty nesters, etc. And the combinations of each with the preferred marketing channel, so you can use these insights for future marketing campaigns. 

3.      Improve supervised learning

Logistic regressions per cluster, this means training one model for each segment of your data to try to improve classification.

#### Common uses of Dimension Reduction in the real world

1. Turn high resolution images into compressed images

This means to come to a reduced, more compact version of those images so they can still contain most of the data that can tell us what the image is about.  

2.  Image tracking

Reduce the noise to the primary factors that are relevant in a video capture. The benefits of reducing the data set can greatly speed up the computational efficiency of the detection algorithms.   

#### K-means Clustering
K-means clustering is an iterative process in which similar observations are grouped together. To do that, this algorithm starts by taking taking 2 random points known as centroids, and starts calculating the distance of each observation to the centroid, and assigning each cluster to the nearest centroid. After the first iteration every point belongs to a cluster.

Next, the number of centroids increases by one, and the centroid for each cluster are recalculated as the points with the average distance to all points in a given cluster. Then we keep repeating this process until no example is assigned to another cluster. 

And this process is repeated k-times, hence the name k-means. This algorithm converges when clusters do not move anymore.

We can also create multiple clusters, and we can have multiple solutions, by multiple solutions we mean that the clusters are not going to move anymore (they converged) but we can converge in different places where we no longer move those centroids.

#### Advantages and Disadvantages of K-Means  

The main advantage of k-means algorithm is that it is easy to compute. One disadvantage is that this algorithm is sensitive to the choice of the initial points, so different initial configurations may yield different results. 

To overcome this, there is a smarter initialization of K-mean clusters called K-means ++, which helps to avoid getting stuck at local optima. This is the default implementation of the K-means.     

#### Model Selection, choosing K number of clusters

Sometimes you want to split your data into a predetermined number of groups or segments. Often, the number of clusters (K) is unclear, and you need an approach to select it.

A common metric is Inertia, defined as the sum of squares distance from each point to its cluster.

Smaller values of Inertia correspond to tighter clusters, this means that we are penalizing spread out clusters and rewarding tighter clusters to their centroids.

The draw back of this metric is that its value sensitive to number of points in clusters. The more points you add, the more you will continue penalizing the inertia of a cluster, even if those points are relatively closer to the centroids than the existing points. 

Another metric is Distortion defined as the average of squared distance from each point to its cluster.

Smaller values of distortion corresponds to tighter clusters.

An advantage, is that distortion doesn’t generally increase as more points are added (relative to inertia). This means that It doesn’t increase distortion, as closer points will aid an actual decreasing the average distance.

#### Inertia Vs. Distortion 

Both are measures of entropy per cluster.

Inertia will always increase as more members are added to each cluster, while this will not be the case with distortion. 

When the similarity of the points in the cluster are very relevan, you should use distortion and if you are more concerned that clusters should have a similar number of points, then you should use inertia.     

#### Finding the right cluster
To find the cluster with a low entopy metric, you run a few k-means clustering models with different initial configurations, comparethe results, and determine which one of the different initializations of configurations lead to the lowest inertia or distortion.

### Distance Metrics
Clustering methods rely very heavily on our definition of distance. Our choice of Distance Metric will be extremely important when discussing our clustering algorithms and to clustering success. 

Each metric has strengths and most appropriate use cases, but sometimes coosing a distance metric is also based on empirical evaluation to determine which metric works best to achieve our goals. 

These are the most common distance metrics:

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/DDABfJewQPSwAXyXsDD0dA_375cff5d8fb0fddf1175c3cb5b1f170a_euclidean.png?raw=true)

#### Euclidean Distance

This one is the most intuitive distance metric, and that we use in K-means, another name for this is the L2 distance. You probably remember from your trigonometry classes.

We calculate (d) by taking the square root of the square of each of this changes (values). We can move this to higher dimensions for example 3 dimensions, 4 dimensions etc.  In general, for an n-dimensional space, the distance is: 


#### Manhattan Distance (L1 or City Block)

Another distance metric is the L1 distance or the Manhattan distance, and instead of squaring each term we are adding up the absolute value of each term. It will always be larger than the L2 distance, unless they lie on the same axis. We use this in business cases where there is very high dimensionality.  

As high dimensionality often leads to difficulty in distinguishing distances between one point and the other, the L1 score does better than the L2 score in distinguishing these different distances once we move into a higher dimensional space. 

#### Cosine Distance

This is a bit less intuitive distance metric. What we really care about the Cosine Distance is the angle between 2 points, for example, for two given points A and B:

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/Wfx_2UBbQ5G8f9lAW9ORaA_502269a7ef5d112cbcfaa0fcb9e8716e_cosine-distance.png?raw=true)

This metric gives us the cosine of the angle between the two vectors defined from the origin to two given points in a two-dimensional space. To translate this definition into higher dimensions, we take the dot product of the vectors and divide it by the norm of each point.

The key to the Cosine distance is that it will remain insensitive to the scaling with respect to the origin, which means that we can move some of the points along the same line and the distance will remain the same. So, any two points on that same array, passing through the origin will have a distance of zero from one another. 

Euclidean VS Cosine distances

-          Euclidean distance is useful for coordinate based measurements.

-          Euclidean distance is more sensitive to curse of dimensionality

-          Cosine is better for data such as text where location of occurrence is less important.   

#### Jaccard Distance

This distance is useful for texts and is often used to word occurrence. 

Consider the following example: 

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/nlCvIkffTMaQryJH35zGIg_0c93b592643e8c2c951f1e4c681a4e93_jaccard_example.jpg?raw=true)

In this case, the Jaccard Distance is going to be one minus the amount of value shared. So, the intersection over that union. This intersection means, the shared values of the two sentences over the length of the total unique values between sentecnes A and B. 

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/hhpple5dTcOaaZXuXZ3DdQ_6994e41013707b91f05cc149407de14f_jaccard_example2.jpg?raw=true)

It can be useful in cases you have text documents and you want to group similar topics together.

#### Hierarchical Clustering
This clustering algorithm, will try to continuously split out and merge new clusters successively until it reaches a level of convergence. 

This algorithm identifies first the pair of points which has the minimal distance and it turns it into the first cluster, then the second pair of points with the second minimal distance will form the second cluster, and so on. As the algorithm continues doing this with all the pairs of closest points, we can turn our points into just one cluster, which is why HAC also needs a stopping criterion.

There are a few linkage types or methods to measure the distance between clusters. these are the most common:

##### Single linkage: minimum pairwise distance between clusters.

It takes the distance between specific points and declare that as the distance between 2 clusters and then it tries to find for all these pairwise linkages which one is the minimum and then we will combine those together as we move up to a higher hierarchy. 

Pros: It helps ensuring a clear separation between clusters.  

Cons: It won’t be able to separate out cleanly if there is some noise between 2 different clusters. 

Complete linkage: maximum pairwise distance between clusters. 

Instead of taking the minimum distance given the points within each cluster, it will take the maximum value. Then from those maximum distances it decides which one is the smallest and then we can move up that hierarchy. 

Pro: It would do a much better job of separating out the clusters if there’s a bit of noise or overlapping points of two different clusters.

Cons: Tends to break apart a larger existing cluster depending on where that maximum distance of those different points may end up lying  

##### Average linkage: Average pairwise distance between clusters. 

Takes the average of all the points for a given cluster and use those averages or clusters centroids to determine the distance between the different clusters. 

Pros: The same as the single and complete linkage. 

Cons: It also tends to break apart a larger existing cluster. 

##### Ward linkage: Cluster merge is based on inertia.

Computes the inertia for all pairs of points and picks the pair that will ultimately minimizes the value of inertia.

The pros and cons are the same as the average linkage. 

Syntax for Agglomerative Clusters
First, import AgglomarativeClustering

From sklearn.cluster import AgglomerativeClustering

then create an instance of class,

agg = AgglomerativeClustering (n_clusters=3, affinity=‘euclidean’, linkage=‘ward’)

and finally, fit the instance on the data and then predict clusters for new data

            agg=agg.fit(X1)

             y_predict=agg.predict(X2)
             
![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/clusters_algorithms_pros_cons.jpg?raw=true)
# Machine Learning Foundation (C) 2020 IBM Corporation
