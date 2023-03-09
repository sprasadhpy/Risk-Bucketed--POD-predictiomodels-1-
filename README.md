## Risk-Bucketed--POD-prediction models-ML

Risk bucketing involves categorizing borrowers based on their creditworthiness into groups that exhibit similar characteristics. The underlying objective of this process is to obtain homogeneous groups or clusters so that credit risk can be estimated more accurately. Failing to distinguish between borrowers with different levels of risk could lead to inaccurate predictions since the model would not be able to capture the distinct characteristics of each group. By dividing borrowers into different groups based on their riskiness, risk bucketing allows for more accurate predictions. Various statistical methods can be used to accomplish this, but we will employ clustering techniques such as K-means and DBScan algorithms to produce homogeneous clusters.

I have implemented DBScan (Density-Based Spatial Clustering of Applications with Noise) clustering algorithm which  is a density-based clustering algorithm that is particularly useful for identifying clusters of arbitrary shape, size, and orientation. Unlike K-Means, DBScan can handle noise and outliers in the data, and it does not require the user to specify the number of clusters beforehand. Instead, it identifies clusters based on the density of the data points in the neighborhood. DBScan is also more robust to the choice of initial cluster centers since it does not depend on randomly initialized centroids.


Notebook: https://github.com/sprasadhpy/Risk-Bucketed--POD-predictiomodels-1-/blob/main/Risk%20Bucketed%20-POD%20Prediction%20ML%20Models.ipynb


## Implemented ML Algorithms : 

***Logistic regression*** is a commonly used classification algorithm in machine learning and data analysis.
It is effective in modeling binary outcomes and predicting the likelihood of an event. This algorithm estimates the probability of a binary outcome using predictor variables.
AUC Curve
![image](https://user-images.githubusercontent.com/40602129/224083547-f51f0d0d-ea9e-45e6-8f54-c52d58b52554.png)


***Bayesian Model :*** To predict the probability of default, the PYMC3 package will be used for Bayesian estimation. Among the several available approaches for Bayesian analysis using PYMC3, the first application will employ the MAP distribution for efficient modeling using the representative posterior distribution. The Bayesian model will also feature a deterministic variable (p) solely dependent on parent variables, including age, job, credit amount, and duration. This comprehensive approach will enable accurate predictions and provide a detailed analysis of the probability of default.
![image](https://user-images.githubusercontent.com/40602129/224084578-51b1b6e5-c2d2-412d-9067-d685d75dd662.png)


Again the implementation enables the user to perform Bayesian analysis, plot the trace, and display the summary statistics for the trace of each logistic model. The implementation logging level for pymc3 to error and defines two logistic models, logistic_model1 and logistic_model2. It then samples from each model using Metropolis as the step method for the sampler and generates the trace. The trace is then plotted using az.plot_trace() and the summary statistics for the trace are displayed using display(az.summary()). 
![image](https://user-images.githubusercontent.com/40602129/224085294-b27f0f1f-ce6b-4fb1-9ce7-e1854e799cac.png)
![image](https://user-images.githubusercontent.com/40602129/224085407-7ae9a8b7-e57e-4b3b-a49f-c710b628e40a.png)



***SupporVectorMachine :*** SVM is known to be a parametric model that performs well with high-dimensional data. It is a suitable approach to use in the case of predicting the probability of default in a multivariate setting. To optimize the performance of SVM and conduct hyperparameter tuning, HalvingRandomSearchCV will be used. This approach utilizes iterative selection and fewer resources, leading to better performance and saving time. HalvingRandomSearchCV employs successive halving to identify candidate parameters by evaluating all parameter combinations with a certain number of training samples in the first iteration, using some of the selected parameters in the second iteration with a larger number of training samples, and finally including only the top-scoring candidates in the model until the last iteration.

![image](https://user-images.githubusercontent.com/40602129/224086244-e40c1177-741a-44b8-aac4-bd4e15392738.png)


***Random Forest :*** The random forest classifier can be used to model the probability of default, and it performs well with large numbers of samples. Using a halving search approach, we can determine the best combination of hyperparameters, including n_estimators, criterion, max_features, max_depth, and min_samples_split.Each cluster has its own set of optimal hyperparameters, and the proposed model is more intricate with a greater depth. Furthermore, the maximum number of features differs across the various clusters.

![image](https://user-images.githubusercontent.com/40602129/224088175-c817cc0c-f5aa-4cf6-9cd8-c212b09a8df0.png)


***XGBoost :*** XGBoost is a boosting algorithm that combines multiple decision trees to create a strong ensemble model. The algorithm iteratively improves the performance of the model by adding new trees that focus on the most challenging examples. 

![image](https://user-images.githubusercontent.com/40602129/224089428-23ffc14a-c4d3-4afe-a01d-68c4ee68cb6e.png)


*** Neural Network:****  To set up the NN model, GridSearchCV optimizes the number of hidden layers, optimization technique, and learning rate. The MLP library controls several parameters, including the size of the hidden layer, the optimization technique (solver), and the learning rate. The optimized hyperparameters of the two clusters differ only in the number of neurons in the hidden layer. Cluster one has a larger number of neurons in the first hidden layer, while cluster two has a larger number in the second hidden layer.

![image](https://user-images.githubusercontent.com/40602129/224089884-c63d9a20-72d1-4526-837b-56aaa72281d9.png)



*** KerasClassifier:****  KerasClassifier enables the use of pre-trained models like CNNs and RNNs for PD estimation, with flexibility in defining network architecture and optimization algorithms. Hyperparameters such as batch size, epoch, and dropout rate can be fine-tuned to specific data needs, while the sigmoid activation function is optimal for classification problems like PD estimation. Deep Learning with NNs provides a complex structure for better predictive performance by capturing the data dynamics.


Best hyperparameters for first cluster in DL are {'batch_size': 10, 'dropout_rate': 0.2, 'epochs': 50}
6/6 [==============================] - 0s 3ms/step
DL_ROC_AUC is 0.5102


Best parameters for second cluster in DL are {'batch_size': 100, 'dropout_rate': 0.4, 'epochs': 150}
2/2 [==============================] - 0s 4ms/step
DL_ROC_AUC is 0.6711





