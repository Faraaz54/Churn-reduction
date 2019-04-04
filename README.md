# Churn-reduction
predicting which customers would churn for a telecom company

# Dataset Description
The predictors provided are as follows:
 ● account length 
 ● international plan 
 ● voicemail plan 
 ● number of voicemail messages 
 ● total day minutes used 
 ● day calls made 
 ● total day charge 
 ● total evening minutes 
 ● total evening calls 
 ● total evening charge 
 ● total night minutes 
 ● total night calls
 ● total night charge 
 ● total international minutes used 
 ● total international calls made
 ● total international charge 
● number of customer service calls made 
Target Variable : move: if the customer has moved (1=yes; 0 = no)
This is an imbalanced dataset as number of customers who churned are high compared to the number of customers who did not Churn(~14%)

# Dataset Preparation
This is an imbalanced data set where the number of samples in each class is not balanced. This would cause a problem to ML models which are classifying our data the models would tend to classify the samples as belonging to the majority class and undermine the minority class samples.
We could address this issue by oversampling our minority class to match the total number of samples in the majority class by randomly sampling from the minority class samples with replacement.
We should also be careful during cross validation as the number of samples in a validation fold should have both the classes represented. This can be addressed by using stratified k-fold during cross validation.

# Evaluation Metric
There are quite a few metrics that need to be considered for our classification task apart from accuracy.
Therefore, we would be analyzing the precision-recall curve, roc curve, auc and f-measure to assess the performance of our classification models
Since, we are oversampling our training samples before fitting our models with the training set or folds during cross-validation; we would be using accuracy as our evaluation metric.
The remaining metrics would be used later on after predicting on our testing set.

# ML Models and Hyper-parameter tuning
We would be using three types of classification models that belong to different families of ML models and therefore, would help us understand how different models perform on our dataset.
Logistic Regression(Generalized Linear Models)
SVM(Non-parametric models)
Gradient-Boosting Trees(Ensemble Models)
We would be performing hyper-parameter tuning of these models using Bayesian optimization which uses the Bayes’ Theorem in the background to assess which parameters would yield a greater score or minimize the loss of an ML model based on past history of parameter values.

# Data Cleaning
# •	Checking for Null Values
The training and the testing set were both tested for null values first. The datasets did not contain any null values.

# •	Dropping Redundant Columns
The phone number column in the training and test set was dropped as it is unique to a customer and would not provide any information for classifying the samples.

# •	Encoding categorical columns
The categorical columns such as Churn, voice-mail plan and international plan contained string values such as True and False or yes and no.
These values where encoded to their numeric representation of 0 and 1 respectively.

# Feature Engineering
We engineered a total of 18 new features from our existing features to give our ML models more scope to differentiate between the two classes.

Some of the engineered features are:
# 1. Voice and international plan
This feature tells us if the customer has opted for both a voice mail and international plan indicating a customer being invested in the telecom company
# 2. Minutes / call
A continuous feature obtained by dividing total day, evening , night and international calls with their respective minutes giving us an idea about minutes spent talking on call at different times of day and internationally.
# 3. Revenue and Revenue per day
These features tell us the total revenue from the customer and revenue per day with respect to the account length.
# 4. Difference from state aggregate
We calculated the median statistic of various features such as number of customer service calls, revenue, day, evening and night calls for different states and obtained a difference of the values from features of customers belonging to their state.
# 5. Statewise churn segments
We engineered a new feature where we segregated each state with respect to the number of customers churned and assigned a number indicating the state belonging to a particular segment.
# 6. Target encoding state and area code
We used target encoding to encode our categorical features state and area code rather than label encoding them with arbitrary numbers. Encoding them this way preserves the distributions the categories share with the Target and provides us with useful values to replace these categories with.

# Conclusion
We started our modelling stage by using Logistic Regression which assumes a linear relationship between the target variable and the predictors and outputs a class probability of a sample belonging to a particular class.

As per the results, we can see that it does a decent job of classifying our test data in terms of accuracy because of the class imbalance as most of the samples belong to the no-churn class.

We need to look at other metrics to gauge the performance of our classifier.
If we observe the F-measure of our churn class, it is quite low which means that it is doing a poor job in correctly classifying our churn class as we can observe the precision and recall of 0.33 and 0.74 respectively. ROC curve also highlights this issue as the AUC is quite low at 0.76 compared to other classifiers used for prediction.

Therefore, we can conclude that there exist non-linearities in our data which cannot be adequately modelled by the logistic regression classifier.

The next classifier we used was the SVM classifier which tries to find a separating plane between the samples in our training data. It also projects our features to a high dimensional space using kernels to find a better separation between classes.

The results show that it was able to capture non linearities in our data going by the metrics calculated from the predictions. It outperforms logistic regression in every evaluation criteria and exhibits a better recall and precision of our churn class.

The final classifier, which is the Gradient Boosting Trees from the family of ensemble models, provides the highest accuracy and the highest precision and recall for our churn class.

High performance of our GBM classifier is attributed to the fact that it combines the predictive power of many weak learners to predict the target class and hence, tends to outperform traditional machine learning models.

