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

### Non Negative Matrix Decomposition  
Non Negative Matrix Decomposition is another way of reducing the number of dimensions. Similar to PCA, it is also a matrix decomposition method in the form V=WxH.  

The main difference is that it can only be applied to matrices that have positive values as inputs, for example:  

pixels in a matrix  
positive attributes that can be zero or higher  
In the case of word and vocabulary recognition, each row in the matrix can be considered a document, while each column can be considered a topic.  

NMF has proven to be powerful for:  

* word and vocabulary recognition  
* image processing,   
* text mining  
* transcribing  
* encoding and decoding  
* decomposition of video, music, or images  
There are advantages and disadvantages of only dealing with non negative values.  

An advantage, is that NMF leads to features that tend to be more interpretable. For example, in facial recognition, the decomposed components match to something more interpretable like, for example, the nose, the eyebrows, or the mouth.  

A disadvantage is that NMF truncates negative values by default to impose the added constraint of only positive values. This truncation tends to lose more information than other decomposition methods.  

Unlike PCA, it does not have to use orthogonal latent vectors, and can end up using vectors that point in the same direction.  

NMF for NLP  
In the case of Natural Language Processing, NMF works as below given these inputs, parameters to tune, and outputs:  

Inputs  

Given vectorized inputs, which are usually pre-processed using count vectorizer or vectorizers in the form Term Frequency - Inverse Document Frequency (TF-IDF).  

Parameters to tune  

The main two parameters are:  

 Number of Topics  
 Text Preprocessing (stop words, min/max document frequency, parts of speech, etc)  
Output  

The output of NMF will be two matrices:  

W Matrix telling us how the terms relate to the different topics.  
H Matrix telling us how to use those topics to reconstruct our original documents.  
 Syntax  
 The syntax consists of importing the class containing the clustering method:  

               from sklearn.decomposition import NMF  

 creating the instance of the class:  

                nmf=NMF(n_components=3, init='random')  

and fit the instance and create a transformed version of the data:  

               x_nmf=NMF.fit(X)
   
### Introduction to Neural Networks
Neural Networks and Deep Learning are behind most of the AI that shapes our everyday life. Think of how you interact everyday with these technologies just by using the greatest features in our phones (face-recognition, autocorrect, text-autocomplete, voicemail-to-text previews), finding what we need on the internet (predictive internet searches, content or product recommendations), or using self-driving cars. Also, some of the classification and regression problems you need to solve, are good candidates for Neural Networks and Deep Learning as well.

#### Some basic facts of Neural Networks:

Use biology as inspiration for mathematical models
Get signals from previous neurons
Generate signals according to inputs
Pass signals on to next neurons
You can create a complex model by layering many neurons
The basic syntax of Multi-Layer Perceptrons in scikit learn is:

   #### Import Scikit-Learn model

              from sklearn.neural_network import MLPClassifier

  #### Specify an activation function

              mlp = MLPClassifier(hidden_layer_sizes=(5,2), activation= 'logistic')

  #### Fit and predict data (similar to approach for other sklearn models)

              mlp.fit(X_train, y_train)

              mlp.predict(X_test)

 

### These are the main parts of MLP:

* Weights
* Input layer
* Hidden Layer
* Weights
* Net Input
* Activation

### Deep Learning Use Cases Summary

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/deep_learning_use_cases_summary.jpg?raw=true)

### Training a Neural Network
In a nutshell this is the process to train a neural network:

Put in Training inputs, get the output.
Compare output to correct answers: Look at loss function J.
Adjust and repeat.
Backpropagation tells us how to make a single adjustment using calculus. 
The vanishing gradient problem is caused due to the fact that as you have more layers, the gradient gets very small at the early layers. For this reason, other activations (such as ReLU) have become more common

The right activation function depends on the application, and there are no hard and fast rules. These are the some of the most used activation functions and their most common use cases:

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/activation_fuctions.jpg?raw=true)

### Deep Learning and Regularization
Technically, a deep Neural Network has 2 or more hidden layers (often, many more). Deep Learning involves Machine Learning with deep Neural Networks. However, the term Deep Learning is often used to broadly describe a subset of Machine Learning approaches that use deep Neural Networks to uncover otherwise-unobservable relationships in the data, often as an alternative to manual feature engineering. Deep Learning approaches are common in Supervised, Unsupervised, and Semi-supervised Machine Learning.

These are some common ways to prevent overfitting and regularize neural networks:

* Regularization penalty in cost function - This option explicitly adds a penalty to the loss function

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/cost_function.jpg?raw=true)

* Dropout - This is a mechanism in which at each training iteration (batch) we randomly remove a subset of neurons. This prevents a neural network from relying too much on individual pathways, making it more robust. At test time the weight of the neuron is rescaled to reflect the percentage of the time it was active.
* Early stopping - This is another heuristic approach to regularization that refers to choosing some rules to determine if the training should stop.
     Example: 

     Check the validation log-loss every 10 epochs.

     If it is higher than it was last time, stop and use the previous model.

* Optimizers - This approaches are based on the idea of tweaking and improving the weights using other methods instead of gradient descent.
  

#### Details of Neural Networks
Training Neural Networks is sensitive to how to compute the derivative of each weight and how to reach convergence. Important concepts that are involved at this step:

Batching methods, which includes techniques like full-batch, mini-batch, and stochastic gradient descent, get the derivative for a set of points

Data shuffling, which aids convergence by making sure data is presented in a different order every epoch.

  

### Keras
Keras is a high-level library that can run on either TensorFlow or Theano. It simplifies the syntax, and allows multiple backend tools, though it is most commonly used with TensorFlow.

This is a common approach to train a deep learning model using Keras:

Compile the model, specifying your loss function, metrics, and optimizer.
Fit the model on your training data (specifying batch size, number of epochs).
Predict on new data.
 
Evaluate your results.
 
Below is the syntax to create a sequential model in Keras.

First, import the Sequential function and initialize your model object:

                 from keras.models import Sequential 

                 model = Sequential()

Then add layers to the model one by one:

               from keras.layers import Dense, Activation 

  

            # Import libraries, model elements

            from keras.models import Sequential 

            from keras.layers import Dense, Activation 

            model = Sequential()

            # For the first layer, specify the input dimension

            model.add(Dense(units=4, input_dim=3)) 

            # Specify activation function

            model.add(Activation('sigmoid')) 

            #For subsequent layers, the input dimension is presumed from the previous layer model.add(Dense(units=4)) 

            model.add(Activation('sigmoid'))


### CNNs
Convolutional Layers have relatively few weights and more layers than other architectures. In practice, data scientists add layers to CNNs to solve specific problems using Transfer Learning.

### Transfer Learning
The main idea of Transfer Learning consists of keeping early layers of a pre-trained network and re-train the later layers for a specific application.

Last layers in the network capture features that are more particular to the specific data you are trying to classify.

Later layers are easier to train as adjusting their weights has a more immediate impact on the final result.

#### Guiding Principles for Fine Tuning

While there are no rules of thumb, these are some guiding principles to keep in mind:

* The more similar your data and problem are to the source data of the pre-trained network, the less intensive fine-tuning will be.
* If your data is substantially different in nature than the data the source model was trained on, Transfer Learning may be of little value.

### CNN Architectures

#### LeNet-5

* Created by Yann LeCun in the 1990s
* Used on the MNIST data set.
* Novel Idea: Use convolutions to efficiently learn features on data set.

#### AlexNet

* Considered the “flash point” for modern deep learning
* Created in 2012 for the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).
* Task: predict the correct label from among 1000 classes.
* Dataset: around 1.2 million images.

AlexNet developers performed data augmentation for training.

* Cropping, horizontal flipping, and other manipulations.

Basic AlexNet Template:

* Convolutions with ReLUs.
* Sometimes add maxpool after convolutional layer.
* Fully connected layers at the end before a softmax classifier.

#### VGG

Simplify Network Structure: has same concepts and ideas from LeNet, considerably deeper.

This architecture avoids Manual Choices of Convolution Size and has very Deep Network with 3x3 Convolutions.

These structures tend to give rise to larger convolutions.

This was one of the first architectures to experiment with many layers (More is better!). It can use multiple 3x3 convolutions to simulate larger kernels with fewer parameters and it served as ”base model” for future works.

#### Inception

Ideated by Szegedy et al 2014, this architecture was built to turn each layer of the neural network into further branches of convolutions. Each branch handles a smaller portion of workload.

The network concatenates different branches at the end. These networks use different receptive fields and have sparse activations of groups of neurons.

Inception V3 is a relevant example of an Inception architecture.

#### ResNet

Researchers were building deeper and deeper networks but started finding these issues:

In theory, the very deep (56-layer) networks should fit the training data better (even if they overfit) but that was not happening.  

Seemed that the early layers were just not getting updated and the signal got lost (due to vanishing gradient type issues).

These are the main reasons why adding layers does not always decrease training error:

* Early layers of Deep Networks are very slow to adjust.
* Analogous to “Vanishing Gradient” issue.
* In theory, should be able to just have an “identity” transformation that makes the deeper network behave like a shallow one.
* In a nutshell, a ResNet:

* Has several layers such as convolutions
* Enforces “best transformation” by adding “shortcut connections”.
* Adds the inputs from an earlier layer to the output of current layer.
* Keeps passing both the the initial unchanged information and the transformed information to the next layer.


### Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are mostly used in applications of natural language processing and speech recognition.

One of the main motivations for RNNs is to derive insights from text and do better than “bag of words” implementations. Ideally, each word is processed or understood in the appropriate context.

Words should be handled differently depending on “context”. Also, each word should update the context.

Under the notion of recurrence, words are input one by one. This way, we can handle variable lengths of text. This means that the response to a word depends on the words that preceded it.

These are the two main outputs of an RNN:

* Prediction: What would be the prediction if the sequence ended with that word.
* State: Summary of everything that happened in the past.

#### Mathematical Details

Mathematically, there are cores and subsequent dense layers

current state = function1(old state, current input).

current output = function2(current state).

We learn function1 and function2 by training our network!  

    r = dimension of input vector

    s = dimension of hidden state

    t = dimension of output vector (after dense layer)

     U is a s × r matrix

    W is a s × s matrix

     V is a t × s matrix

In which the weight matrices U, V, W are the same across all positions

#### Practical Details

Often, we train on just the ”final” output and ignore intermediate outputs.

Slight variation called Backpropagation Through Time (BPTT) is used to train RNNs.

Sensitive to length of sequence (due to “vanishing/exploding gradient” problem).

In practice, we still set a maximum length to our sequences. If the input is shorter than maximum, we “pad” it. If the input is longer than maximum, we truncate it.

#### RNN Applications

RNNs often focus on text applications, but are commonly used for other sequential data:

* Forecasting: Customer Sales, Loss Rates, Network Traffic.  
* Speech Recognition: Call Center Automation, Voice Applications. 
* Manufacturing Sensor Data 
* Genome Sequences

#### Long-Short Term Memory RNNs (LSTM)

LSTMs are a special kind of RNN (invented in 1997). LSTM has as motivation solve one of the main weaknesses of RNNs, which is that its transitional nature, makes it hard to keep information from distant past in current memory without reinforcement.

LSTM have a more complex mechanism for updating the state.

Standard RNNs have poor memory because the transition Matrix necessarily weakens signal.

This is the problem addressed by Long-Short Term Memory RNNs (LSTM).

To solve it, you need a structure that can leave some dimensions unchanged over many steps.

* By default, LSTMs remember the information from the last step.
* Items are overwritten as an active choice.

The idea for updating states that RNNs use is old, but the available computing power to do it sequence to sequence mapping, explicit memory unit, and text generation tasks is relatively new.

Augment RNNs with a few additional Gate Units:

* Gate Units control how long/if events will stay in memory.
* Input Gate: If its value is such, it causes items to be stored in memory.
* Forget Gate: If its value is such, it causes items to be removed from memory.
* Output Gate: If its value is such, it causes the hidden unit to feed forward (output) in the network.

#### Gated Recurrent Units (GRUs)

GRUs are a gating mechanism for RNNs that is an alternative to LSTM. It is based on the principle of Removed Cell State:

* Past information is now used to transfer past information.
* Think of as a “simpler” and faster version of LSTM.

These are the gates of GRU:

Reset gate: helps decide how much past information to forget.

Update gate: helps decide what information to throw away and what new information to keep.

#### LSTM vs GRU

LSTMs are a bit more complex and may therefore be able to find more complicated patterns.

Conversely, GRUs are a bit simpler and therefore are quicker to train.

GRUs will generally perform about as well as LSTMs with shorter training time, especially for smaller datasets.

In Keras it is easy to switch from one to the other by specifying a layer type. It is relatively quickly to change one for the other.

#### Sequence-to-Sequence Models (Seq2Seq)

Thinking back to any type of RNN interprets text, the model will have a new hidden state at each step of the sequence containing information about all past words.

Seq2Seq improve keeping necessary information in the hidde state from one sequence to the next.

This way, at the end of a sentence, the hidden state will have all information relating to past words. 

The size of the vector from the hidden state is the same no matter the size of the sentence.

In a nutshell, there is an encoder, a hidden state, and a decoder.

#### Beam Search

Beam search is an attempt to solve greedy inference.

* Greedy Inference, which means that a model producing one word at a time implies that if it produces one wrong word, it might output a wrong entire sequence of words.
* Beam search tries to produce multiple different hypotheses to produce words until <EOS> and then see which full sentence is most likely.
These are examples of common enterprise applications of LSTM models:

* Forecasting: (LSTM among most common Deep Learning models used in forecasting).
* Speech Recognition
* Machine Translation
* Image Captioning
* Question Answering
* Anomaly Detection
* Robotic Control

#### Autoencoders

Autoencoders are a neural network architecture that forces the learning of a lower dimensional representation of data, commonly images.

Autoencoders are a type of unsupervised deep learning model that use hidden layers to decompose and then recreate their input. They have several applications:

* Dimensionality reduction 
* Preprocessing for classification
* Identifying ‘essential’ elements of the input data, and filtering out noise
One of the main motivations is find whether two pictures are similar.

#### Autoencoders and PCA

Autoencoders can be used in cases that are suited for Principal Component Analysis (PCA).

Autoencoders also help to deal with some of these PCA limitations: PCA has learned features that are linear combinations of original features.

Autoencoders can detect complex, nonlinear relationship between original features and best lower dimensional representation.

#### Autoencoding process

The process for autoencoding can be summarized as:

* Feed image through encoder network
* Generate the lower dimension embedding
* Feed embedding through decoer network
* Generate reconstructed version of the original data
* Compare the result of the generated vs the original image

Result: A network will learn the lower dimensional space that represents the original data

#### Autoencoder applications

Autoencoders have a wide variety of enterprise applications:

* Dimensionality reduction as preprocessing for classification
* Information retrieval
* Anomaly detection
* Machine translation
* Image-related applications (generation, denoising, processing and compression)
* Drug discovery
* Popularity prediction for social media posts
* Sound and music synthesis
* Recommender systems

#### Variational Autoencoders

Variational autoencoders also generate a latent representation and then use this representation to generate new samples (i.e. images). 

These are some important features of variational autoencoders:

* Data are assumed to be represented by a set of normally-distributed latent factors.
* The encoder generates parameters of these distributions, namely µ and σ.
* Images can be generated by sampling from these distributions.

#### VAE goals

The main goal of VAEs: generate images using the decoder

The secondary goal is to have similar images be close together in latent space

#### Loss Function of Variational Autoencoders

The VAE reconstruct the original images from the space of a vector drawn from a standard normal distribution.

The two components of the loss function are:

* A penalty for not reconstructing the image correctly. 
* A penalty for generating vectors of parameters µ and σ that are different than 0 and 1, respectively: the parameters of the standard normal distribution.

### Generative Adversarial Networks (GANs)
The invention of GANs was connected to neural networks’ vulnerability to adversarial examples. Researchers were going to run a speech synthesis contest, to see which neural network could generate the most realistic-sounding speech.

A neural network - the “discriminator” - would judge whether the speech was real or not.

In the end, they decided not to run the contest, because they realized people would generate speech to fool this particular network, rather than actually generating realistic speech.

These are the step to train GANs

* Randomly initialize weights of generator and discriminator networks
* Randomly initialize noise vector and generate image using generator
* Predict probability generated image is real using discriminator
* Compute losses both assuming the image was fake and assuming it was real
* Train the discriminator to output whether the image is fake
* Compute the penalty for the discriminator probability, without using it to train the discriminator
* Train the generator to generate images that the discriminator thinks are real
* Use the discriminator to calculate the probability that a real image is real
* Use L to train the discriminator to output 1 when it sees real images

### Reinforcement Learning
In Reinforcement Learning, Agents interact with an Environment

They choose from a set of available Actions

The actions impact the Environment, which impacts agents via Rewards

Rewards are generally unknown and must be estimated by the agent

The process repeats dynamically, so agents learn how to estimate rewards over time

Advances in deep learning have led to many recent RL developments:

* In 2013, researchers from DeepMind developed a system to play Atari games
* In 2017, the AlphaGo system defeated the world champion in Go
In general, RL algorithms have been limited due to significant data and computational requirements.

As a result, many well-known use cases involve learning to play games. More recently, progress has been made in areas with more direct business applications.

### Reinforcement Learning Architecture
The main components of reinforcement learning are: Policy, Agents, Actions, State, and Reward.

Solutions represents a Policy by which Agents choose Actions in response to the State

Agents typically maximize expected rewards over time

In Python, the most common library for RL is Open AI GYM

This differs from typical Machine Learning Problems:

Unlike labels, rewards are not known and are often highly uncertain

As actions impact the environment, the state changes, which changes the problem

Agents face a tradeoff between rewards in different periods

Examples of everyday applications of Reinforcement Learning include recommendation engines, marketing, and automated bidding.

### Time Series

Time Series is a sequence of data points organized in time order.

The sequence captures data at equally spaced points in time. Data that is not collected regularly at equally spaced points is not considered time series.

#### Time Series Motivation
For most forecasting exercises, standard regression approaches do not work for Time Series models, mostly because:

* The data is correlated over time
* The data is often non-stationary, which is hard to model using regressions
* You need a lot of data for a forecast

#### Forecasting Problems

These are the two types of forecasting problems. Consider that the vast majority of applications employ univariate models, harder to combine variables when using time series data.

1.      Univariate

Think of single data series containing of:

* Continuous data, binary data, or categorical data
* Multiple unrelated series 
* Conditional series

2.      Panel or Multivariate

Think of multiple related series identifying groups such as customer types, department or channel, or geographic joint estimation across series

#### Time Series Applications
Time series data is common across many industries. For example:

* Finance: stock prices, asset prices, macroeconomic factors
* E-Commerce: page views, new users, searches
* Business: transactions, revenue, inventory levels 

Time series methods are used to:

* Understand the processes driving observed data
* Fit models to monitor or forecast a process
* Understand what influences future results of various series
* Anticipate events that require management intervention

#### Time Series Components

A time series can be decomposed into several components:

            Trend – long term direction

            Seasonality – periodic behavior

            Residual – irregular fluctuations

Generally, models perform better if we can first remove known sources of variation such as trend and seasonality. The main motivation for doing decomposition is to improve model performance. Usually we try to identify the known sources and remove them, leaving resulting series (residuals) that we can fit against time series models

Decomposition Models
These are the main models to decompose Time Series components:

            –       Additive Decomposition Model
Additive models assume the observed time series is the sum of its components.

            i.e. Observation = Trend + Seasonality + Residual

These models are used when the magnitudes of the seasonal and residual values are independent of trend.

            –       Multiplicative Decomposition Model
Multiplicative models assume the observed time series is the product of its components.

            i.e. Observation = Trend * Seasonality * Residual

A multiplicative model can be transformed to an additive by applying a log transformation:

            log(Time*Seasonality*Residual) = log(Time) + log(Seasonality) + log(Residual)

These models are used if the magnitudes of the seasonal and residual values fluctuate with trend.

            –       Pseudo-additive Decomposition Model
Pseudo-additive models combine elements of the additive and multiplicative models.

They can be useful when:

Time series values are close to or equal to zero.

We expect features related to a multiplicative model.

A division by zero needs to be solved in the form:
            
            Ot = Tt + Tt(St – 1) + Tt(Rt – 1) = Tt(St + Rt – 1) 

Decomposition of time series allows us to remove deterministic components, which would otherwise complicate modeling.

After removing these components, the main focus is to model the residual.

Other Methods
These are some other approaches of time series decomposition: 

* Exponential smoothing
* Locally Estimated Scatterplot Smoothing (LOESS)
* Frequency-based methods

### Stationarity

Stationarity impacts our ability to model and forecast

* A stationary series has the same mean and variance over time
* Non-stationary series are much harder to model

Common approach:

* Identify sources of non-stationarity
* Transform series to make it stationary
* Build models with stationary series

The Augmented Dickey-Fuller (ADF) test specifically tests for stationarity.

* It is a hypothesis test: the test returns a p-value,  and we generally say the series is non-stationary if the p-value is less than 0.05.
* It is a less appropriate test to use with small datasets,  or data with heteroscedasticity (different variance across observations) present.
* It is best to pair ADF with other techniques such as:  run-sequence plots, summary statistics, or histograms.
Common Transformations for Time Series include:

Transformations allow us to generate stationary inputs required by most models.

There are several ways to transform nonstationary time series data:

* Remove trend (constant mean)
* Remove heteroscedasticity with log (constant variance)
* Remove autocorrelation with differencing (exploit constant structure)
* Remove seasonality (no periodic component)
* Multiple transformations are often required.

### Time Series Smoothing
Smoothing is a process that often improves our ability to forecast series by reducing the impact of noise.

There are many ways to smooth data. Some examples:

* Simple average smoothing
* Equally weighted moving average
* Exponentially weighted moving average

This are some suggestions for selecting a Smoothing Technique. If your data:

            –       lack a trend
            Then use Single Exponential Smoothing

            –       have trend but no seasonality
            Then use Double Exponential Smoothing

            –       have trend and seasonality
            Then use Triple Exponential Smoothing
            
### ARMA Models

ARMA models combine two models:

* The first is an autoregressive (AR) model. Autoregressive models anticipate series’ dependence on its own past values.
* The second is a moving average (MA) model. Moving average models anticipate series’ dependence on past forecast errors.
* The combination (ARMA) is also known as the Box - Jenkins approach.

ARMA models are often expressed using orders p and q for the AR and MA components. 

For a time series variable X that we want to predict for time t, the last few observations are:


![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/ARMA_models_1.jpg?raw=true)
 

AR(p) models are assumed to depend on the last p values of the time series. For p=2, the forecast has the form:

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/ARMA_models_2.jpg?raw=true)

MA(q) models are assumed to depend on the last q values of the forecast error. For q=2, the forecast has the form:

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/ARMA_models_3.jpg?raw=true)

Combining the AR(p) and MA(q) models yields the ARMA(p, q) model. For p=2, q=2, the ARMA(2, 2) forecast has the form:

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/ARMA_models_4.jpg?raw=true)

#### ARMA Models Considerations
These are important considerations to keep in mind when dealing with ARMA models:

* The time series is assumed to be stationary.
* A good rule of thumb is to have at least 100 observations when fitting an ARMA model.

There are three stages in building an ARMA model:

#### Identification

At this stage you:

* Validate that the time series is stationary.  
* Confirm whether the time series contains a seasonal component.  
You can determine if seasonality is present by using autocorrelation and partial autocorrelation plots, seasonal subseries plots, and intuition (possible in some cases, i.e. seasonal sales of consumer products, holidays, etc.).

An Autocorrelation Plot is commonly used to detect dependence on prior observations. 

It summarizes total (2-way) correlation between the variable and its past values.

The Partial Autocorrelation Plot also summarizes dependence on past observations. 

However, it measures partial results (including all lags)

Seasonal Subseries Plot is one approach for measuring seasonality. This chart shows the average level for each seasonal period and illustrates how individual observations relate to this level.

#### Estimation

Once we have a stationary series, we can estimate AR and MA models. We need to determine p and q, the order of the AR and MA models. 

One approach here is to look at autocorrelation and partial autocorrelation plots. Another approach is to treat p and q as hyperparameters and apply standard approaches (grid search, cross validation, etc.)

How do we determine the order p of the AR model?

* Plot confidence intervals on the Partial Autocorrelation Plot.  
* Choose lag p such that partial autocorrelation becomes insignificant for p + 1 and beyond  

How can we determine the order q of the MA model?  

* Plot confidence intervals on the Autocorrelation Plot  
* Choose lag q such that autocorrelation becomes insignificant for q + 1 and beyond.  

#### Evaluation

You can assess your ARMA model by making sure that the residuals will approximate a Gaussian distribution (aka white noise). Otherwise, you need to iterate to obtain a better model.

These are guidelines to choose between an AR and a MA model based on the shape of the autocorrelation and partial autocorrelation plots.

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/evaluation_table.jpg?raw=true)

#### ARIMA Models

ARIMA stands for Auto-Regressive Integrated Moving Average. 

ARIMA models have three components:

* AR Model
* Integrated Component
* MA Model

#### SARIMA Models

SARIMA is short for Seasonal ARIMA, an extension of ARIMA models to address seasonality.

This model is used to remove seasonal components.

* The SARIMA model is denoted SARIMA (p, d, q) (P, D, Q).
* P, D, Q represent the same as p, d, q but they are applied across a season.
* M = one season
  

#### ARIMA and SARIMA Estimation  
These are the steps to estimate p, d, q and P, D, Q?

* Visually inspect a run sequence plot for trend and seasonality.
* Generate an ACF Plot.
* Generate a PACF Plot.
* Treat as hyperparameters (cross validate).
* Examine information criteria (AIC, BIC) which penalize the number of parameters the model uses.


### Deep Learning for Time Series Forecasting
Neural networks offer several benefits over traditional time series forecasting models, including:

* Automatically learn how to incorporate series characteristics like trend, seasonality,  and autocorrelation into predictions.
Able to capture very complex patterns.
* Can simultaneously model many related series instead of treating each separately.

Some disadvantages of using Deep Learning for Time Series Forecasting are:

* Models can be complex and computationally expensive to build (GPUs can help).
* Deep Learning models often overfit.
* It is challenging to explain / interpret predictions made by the model (“black box”).
* Tend to perform best with large training datasets.

Recurrent neural networks (RNNs) map a sequence of inputs to predicted output(s).

* Most common format is “many-to-one”, that maps an input sequence to one output value.
* Input at each time step sequentially updates the RNN cell’s “hidden state” (“memory”).
* After processing the input sequence, the hidden state information is used to predict the output.

RNNs often struggle to process long input sequences. It is mathematically difficult for RNNs to capture long-term dependencies over many time steps, which is a problem for Time Series, as sequences are often hundreds of steps. Another type of Neural Networks, Long short-term memory networks (LSTMs) can mitigate these issues with a better memory system

Long short-term memory networks share RNNs’ conceptual structure.

* LSTM cells have the same role as RNN cells in sequential processing of the input sequence.
* LSTM cells are internally more complex, with gating mechanisms and two states: a hidden state and a cell state.

Long short-term memory networks regulate information flow and memory storage.

* LSTM cells share forget, input, and output gates that control how memory states are updated and information is passed forward.
* At each time step, the input and current states determine the gate computations. 

#### LSTMs vs RNNs

LSTMs are better suited for handling long-term dependencies than RNNs. However, they are much more complex, requiring many more trainable weights. As a result, LSTMs tend to take longer to train (slower backpropagation) and can be more prone to overfitting.

These are some guidelines on how to choose LSTMs or RNNs in a Forecasting task:

Always consider the problem at hand:

* If sequences are many time steps long, an RNN may perform poorly.
* If training time is an issue, using a LSTM may be too cumbersome.
* Graphics processing units (GPUs) speed up all neural network training,  but are especially recommended when training LSTMs on large datasets.

#### Survival Analysis

Survival Analysis focuses on estimating the length of time until an event occurs. It is called ‘survival analysis’ because it was largely developed by medical researchers interested in estimating the expected lifetime of different cohorts. Today, these methods are applied to many types of events in the business domain.

Examples:

* How long will a customer remains on books before churning
* How long until equipment needs repairs
Survival Analysis is useful when we want to measure the risk of events occurring and our data are Censored. 

* This can be referred to as failure time, event time, or survival time.
* If our data are complete and unbiased, standard regression methods may work.
* Survival Analysis allows us to consider cases with incomplete or censored data.

The Survival Function is defined as S(t)=P(T>t)S(t)=P(T>t) . Itmeasures the probability that a subject will survive past time t.

This function:

* Is decreasing (non-increasing) over time.
* Starts at 1 for all observations when t=0
* Ends at 0 for a high-enough t

The Hazard Rate is defined as: 

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/hazard_rate1.jpg?raw=true)
  
It represents the instantaneous rate at which events occur, given that it has not occurred already.

The cumulative hazard rate (sum of h(t)h(t) from t = 0  to t = t) represents accumulated risk over time.

The Kaplan-Meier estimator is a non-parametric estimator. It allows us to use observed data  to estimate the survival distribution. The Kaplan-Meier Curve plots the cumulative probability of survival beyond each given time period.

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/survival_probability.jpg?raw=true)

Using the Kaplan-Meier Curve allows us to visually inspect differences in survival rates by category. We can use Kaplan-Meier Curves to examine whether there appear to be differences based on this feature.

To see whether survival rates differ based on number of services, we estimate Kaplan-Meier curves for different groups.

  

#### Survival Analysis Approaches

The Kaplan-Meier approach provides sample averages. However, we may want to make use of individual-level data to predict survival rates.

Some well-known Survival models for estimating Hazard Rates include these Survival Regression approaches. These methods:

* Allow us to generate estimates of total risk as a function of time
* Make use of censored and uncensored observations to predict hazard rates
* Allow us to estimate feature effects
Although these methods use time, these methods are not generally predicting a time to an event, rather predicting survival risk (or hazard risk) as a function of time.

            –       The Cox Proportional Hazard (CPH) model 
This is one of the most common survival models. It assumes features have a constant proportional impact on the hazard rate. 

For a single non-time-varying feature X, the hazard rate h(t) is modeled as:

![alt text](https://github.com/KonuTech/IBM-Machine-Learning/blob/main/hazard_rate.jpg?raw=true)

β(t) is the time-varying baseline hazard, and e^(βX) is the (constant) proportional adjustment to the baseline hazard due to X.

Using the CPH model, we can plot estimated survival curves for various categories. 

            –       Accelerated Failure Time (AFT) models (several variants including the Weibull AFT model)
            These models differ with respect to assumptions they make about the hazard rate function, and the impact of features. 

# Machine Learning Foundation (C) 2020 IBM Corporation
