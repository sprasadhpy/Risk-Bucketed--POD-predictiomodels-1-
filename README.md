## Risk-Bucketed--POD-prediction models-ML

Risk bucketing involves categorizing borrowers based on their creditworthiness into groups that exhibit similar characteristics. The underlying objective of this process is to obtain homogeneous groups or clusters so that credit risk can be estimated more accurately. Failing to distinguish between borrowers with different levels of risk could lead to inaccurate predictions since the model would not be able to capture the distinct characteristics of each group. By dividing borrowers into different groups based on their riskiness, risk bucketing allows for more accurate predictions. Various statistical methods can be used to accomplish this, but we will employ clustering techniques such as K-means and DBScan algorithms to produce homogeneous clusters.

I have implemented DBScan (Density-Based Spatial Clustering of Applications with Noise) clustering algorithm which  is a density-based clustering algorithm that is particularly useful for identifying clusters of arbitrary shape, size, and orientation. Unlike K-Means, DBScan can handle noise and outliers in the data, and it does not require the user to specify the number of clusters beforehand. Instead, it identifies clusters based on the density of the data points in the neighborhood. DBScan is also more robust to the choice of initial cluster centers since it does not depend on randomly initialized centroids.


Implemented ML Algorithms : 

Logistic regression is a commonly used classification algorithm in machine learning and data analysis.
It is effective in modeling binary outcomes and predicting the likelihood of an event. This algorithm estimates the probability of a binary outcome using predictor variables.
AUC Curve
![image](https://user-images.githubusercontent.com/40602129/224083547-f51f0d0d-ea9e-45e6-8f54-c52d58b52554.png)


Bayesian Model : To predict the probability of default, the PYMC3 package will be used for Bayesian estimation. Among the several available approaches for Bayesian analysis using PYMC3, the first application will employ the MAP distribution for efficient modeling using the representative posterior distribution. The Bayesian model will also feature a deterministic variable (p) solely dependent on parent variables, including age, job, credit amount, and duration. This comprehensive approach will enable accurate predictions and provide a detailed analysis of the probability of default.
![image](https://user-images.githubusercontent.com/40602129/224084578-51b1b6e5-c2d2-412d-9067-d685d75dd662.png)


Again the implementation enables the user to perform Bayesian analysis, plot the trace, and display the summary statistics for the trace of each logistic model. The implementation logging level for pymc3 to error and defines two logistic models, logistic_model1 and logistic_model2. It then samples from each model using Metropolis as the step method for the sampler and generates the trace. The trace is then plotted using az.plot_trace() and the summary statistics for the trace are displayed using display(az.summary()). 
![image](https://user-images.githubusercontent.com/40602129/224085294-b27f0f1f-ce6b-4fb1-9ce7-e1854e799cac.png)
![image](https://user-images.githubusercontent.com/40602129/224085407-7ae9a8b7-e57e-4b3b-a49f-c710b628e40a.png)



