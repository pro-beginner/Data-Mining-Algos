Following are the instructions for the algorithms

<h2> Linear Regression</h2>

Inputs : The file data0.txt containing data to perform linear regression

Output : A set of weights w, that enable the best linear prediction for the data.

STEPS:
1. Separate the data into two component X, and Y. Marks the fact that X is a biased set of attributes, with a bias attribute
   containing all ones.
2. Any line using the independent variable x is of the form w0 + w1x.
3. Therefore the prediction for the yi the values is obtained as w0 + w1xi.
4. Consequently the total error while predicting all the ys is given as J(w) = ∑(yi−xTiw)2
5. In terms of matrix multiplication this can also be written as (Y - Xw)T(Y-Wx).
6. Since the error is to be minimized, the derivative of this should equal zero.
7. The derivative of the expression in Step. 6 is given as XT(Y-Xw).
8. Find the set of w that produces the best linear prediction.
9. Plot the original values and the predicted values on the same plot.

HINTS: Vectors in numpy are not treated as one dimensional matrices by default. You will have to manually reshape them in to the desired shape. Use numpy.reshape for the same. Numpy does matrix inverses using the submodule numpy.linalg

<h2> Logistic Regression </h2>

Logistic Regression Classifier Perform Logistic Regression on the IRIS dataset in a One vs All (OvA) manner:

Aim: Write a classifier based on the principles of Regression Analysis that identifies a binary classification pattern. The classifier takes the form of a hyper plane in an attribute dimensional space and has the form w0+w1x1+w2x2+...+wmxm i.e. ∑mj=0wjxj WARNING: Watch out for matrix dimensions when multiplying.

Method:

1. Obtain a normalized copy of the iris dataset. The dataset contains two components X the attribute matrix and Y the label vector.
2. Ensure the target class (say "setosa") is converted to a one while all other classes are converted to a zero. We now have a "Binary Classification Problem."
3. Conventionally in ML problems we treat all vectors as column vectors. So each tuple is to be in the form of a column. This is easily obtained by transposing the matrix X.
5. Augment the data matrix by adding a cosmetic attribute which contains the value 1 for all tuples. This is the bias attribute to compensate for w0.
6. Optionally decompose your data into a train and test set.
7. Obtain an initial weight matrix containing "m+1" weights i.e. W=[w0,w1,w2,...wm]T. (Why m+1 when there are only m attributes?)
8. While not converged: a. Make your predictions as H=sigmoid(WTX), where the multiplication is a matrix mul, and sigmoid is defined as g(x)=\frac{1}{1+e^{-x}). (Mark the fact that the matrix multiplication generates the eq. of a plane in exactly the form that is required.) b. Calculate the penalty for the current set of weights as J(W)=−1N∑Ni=1[y(i)log(h(i))+(1−y(i))log(1−h(i))] (Can you justify the choice of the penalty function?) c. Calculate the error in each iteration and store in a buffer. d. Update the weights using the gradient descent strategy as wnewj=woldj−α∇J(W) where ∇J(W)=−(y(i)−h(i))x(i)j. Nabla is nothing but the derivative of the error function J. (BE VERY CAREFUL WITH THE SIGNS.) Alpha is a very small constant known as the learning rate, usually held at 0.01.
9. Make your final predictions with the converged set of weights as follows: h(i)=1 if sigmoid(WTX)≥0.5 else 0.
10. What is the accuracy of your classifier? How does it vary when you change the number of generations. Also plot error against generation and validate the results.
11. Use only two attributes to make a prediction and plot a prediction line showing the separation of the flowers in a scatter plot. EXTENSIONS How can we write the program so that it can calculate three set of weights for the three distinct classes at the same time. Then your prediction for an unknown flower would be max[p(class1),p(class2),p(class3)].

Often it is advisable to regularise weights so as to prevent overfitting. In this case the error function is obtained as J(W)=J(W)+1M∑Mj=0w2j. How would you update the weights if regulariaztion is to be applied to all the weights except the bias i.e. wnew0 is updated using the old formula but for all other weights a new update formula is required.

<h2> Naive Bayesian Clasifier </h2>

INPUT: The dataset consisting of data attributes and a class.

OUTPUT: A Classifier Model

STEPS:

1. Initially we assume that the dataset is comprised of only categorical attributes. There are "N" tuples, "C" classes and "M" attributes.
2. Our task is to find for an unknown tuple t, the probability P(Cj | t) for all Cj. We can then assign the class with the highest probability as the predicted class.
3. From Bayesian probability P(Cj | t) = P(t | Cj)*P(Cj) / P(t). (Mark the fact the denominator is a constant for all Cj).
In the naive case we assume the probability of a tuple is the product of the probabilities of the individual domain values for each of the attributes.
4. The result will often require the Laplace correction to be enforced for viability of the product operation.
5. In case the products become very small the multiplication can be replaced by taking the natural log of the products which transpires to the addition of the natural logs.
6. Find the accuracy of the model that you have built on the test set. (Expected accuracy ~75%)

How can we make the algorithm work for numeric attributes ?

Hints: We can use the sklearn cross validation module to perform a train set / test set split.

<h2> K Means </h2>


Before starting off with the exercise please practice using numpy's where and delete functions. You should also explore the 3D plotting interface for matplotlib.

PERFORM K-Means Clustering (BASIC VERSION)

INPUT:

The data set to be clustered (assumed to have "M" numeric attributes and "N" data points), which is assumed to be without a class label attribute. The number of clusters to create STEPS:

  1. Start by loading the iris dataset and normalizing it with the Standard Scaler.
  2. Initialize "k" (where k is the number of clusters reqd.) centers each having "M" attributes.
  3. Initialize an assignment array of size "N"
  4. For each data point in the dataset: a. Compute the distances from all the centers. b. Sort the distances. c. Find the nearest cluster center. d. Update the assignment array with the nearest cluster center for the current point.
  5. Recompute all the cluster centers by computing the mean of all the points which have been clustered with a particular cluster center.
  6. Go back to step 4.
  7. Continue till no changes are found in the assignment array.
  8. Compute the correctness of your cluster using the homogeneity metrics found in scikit learn's metrics sub module.

WHAT ARE THE ISSUES THAT YOU FIND ? DID WE OBTAIN THE REQUISITE NUMBER OF CLUSTERS EVERY TIME?

SOLUTIONS ??


<h2> K-NN Classifier </h2>
TASK: Perform K-NN Classification on the Iris dataset (K is an odd integer)

THIS IS AN EXAMPLE OF A LAZY LEARNER CLASSIFIER WHERE WE DO NOT BUILD A CLASSIFIER MODEL FIRST. RATHER WHAT EVER DATA IS AVAILABLE IS USED AS AND WHEN AN UNTESTED TUPLE BECOMES AVAILABLE.

Step 1: Normalize the data using Z-norm.

Step 2: Partition the data and the class labels into two sets (called the TRAIN SET and the TEST SET). The train set should contain approximately 75% of the tuples in the dataset and the corresponding labels. The remaining 25% should be part of the test set.

STEP 3: For each tuple in the test set predict the class label according to the following algorithm. a) Find the distance to all points in the TRAIN SET form the current tuple. b) Find the K nearest train tuples. c) Find the majority class label among the K nearest tuple. e) Assign the majority label as the prediction for the current test tuple. f) perform the same for all test tuples.

STEP 4: From your prediction array, the array containg all your predictions, find the accuracy of the system. You can do this by finding the percentage of tuples where your prediction matched the actual class label.

STEP 5: Vary the value of K and plot a graph for the accuracy of the system for different values of K.

STEP 6: TO BE DONE AS AN ASSIGNMENT IN SPARE TIME Use PCA to reduce the number of features to "M" and repeat the above experiment. Plot a curve of the accuracy in this case by varying K.

HINTS: numpy.argsort may be necessary.


<h2> HI AGNES </h2>
HIerarchical Agglomerative Clustering (AGNES) INPUTS: X - A Dataset of dimensions N x M k - The number of dimensions required

Method:

  1. Start off by assuming every data point is a cluster of its own. (So you have N clusters initially).
  2. Generate a distance matrix for all remaining clusters (This is time consuming)
  3. While no_of_clusters_remaining > k: a. Find the two nearest remaining clusters. b. Merge them and update the center for the merged cluster. (Merging entails bringing the current clusters together in a new cluster, deleting one of the clusters and then updating the center of the new cluster as the mean of the two clusters you have just brought together. Hint LISTS in Python can be made up of other lists). c. Update the distance matrix. (You have one cluster less now)

OUTPUT: The final cluster centers Observations: Do you think this is a slow approach. Why?

What about the validity of the results.


<h2> Adaline NN </h2>

LEARN AN ADALINE NN:

INPUT: Any two attributes of the iris dataset. The class labels with any one target class dicretized to 1 others as 0.

OUTPUT: The line of separation obtained finally. Final weights. Final accuracy

Method:

1. Decide on the number of epochs to use:
2. Start with a random weight vector (M+1, 1) (M is the number of attributes,one is for the bias);
3. For each epoch: For each training set data: a. Predict h(i) = summation of w(i)x(i) for all inputs using bias. b. g(i) = sigmoid(h(i)). c. err = (t(i) - g(i)) (t(i) is the known class label for the ith tuple) d. w(j)(new) = w(j)(old) + learning_rateerrx(i) (remember j is for weights while i is for tuples) e. Update the weights
4. Make the final predictions.
5. Report the accuracy.
6. Plot the separation line on the scatter plot.

EXTRA CREDIT: Generalize the algorithm to learn the weights for all the classes simultaneoulsy.

