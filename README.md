## Risk-Bucketed--POD-prediction models-ML

Risk bucketing involves categorizing borrowers based on their creditworthiness into groups that exhibit similar characteristics. The underlying objective of this process is to obtain homogeneous groups or clusters so that credit risk can be estimated more accurately. Failing to distinguish between borrowers with different levels of risk could lead to inaccurate predictions since the model would not be able to capture the distinct characteristics of each group. By dividing borrowers into different groups based on their riskiness, risk bucketing allows for more accurate predictions. Various statistical methods can be used to accomplish this, but we will employ clustering techniques such as K-means and DBScan algorithms to produce homogeneous clusters.

I have implemented DBScan (Density-Based Spatial Clustering of Applications with Noise) clustering algorithm which  is a density-based clustering algorithm that is particularly useful for identifying clusters of arbitrary shape, size, and orientation. Unlike K-Means, DBScan can handle noise and outliers in the data, and it does not require the user to specify the number of clusters beforehand. Instead, it identifies clusters based on the density of the data points in the neighborhood. DBScan is also more robust to the choice of initial cluster centers since it does not depend on randomly initialized centroids.


Implemented ML Algorithms : 

-Logistic regression is a commonly used classification algorithm in machine learning and data analysis.
It is effective in modeling binary outcomes and predicting the likelihood of an event. This algorithm estimates the probability of a binary outcome using predictor variables.
AUC Curve
![image](https://user-images.githubusercontent.com/40602129/224083547-f51f0d0d-ea9e-45e6-8f54-c52d58b52554.png)




