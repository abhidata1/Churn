# Telco Customer Churn Analysis
#Overview
This repository provides an end-to-end machine learning workflow to predict customer churn in a telecommunications company using Databricks. The notebook includes data ingestion, preprocessing, feature engineering, model training, evaluation, and visualization of results. It demonstrates a comparison of multiple classification models including Logistic Regression, Random Forest, Support Vector Machines, k-Nearest Neighbors, and Gradient Boosting.

#Files and Structure
notebook.ipynb: The main Databricks notebook where all the analysis and modeling are performed.
README.md: A file explaining the purpose and structure of the repository.
data/: Directory (not included here) to hold the WA_Fn-UseC_-Telco-Customer-Churn.csv dataset. Make sure the dataset is uploaded to your Azure Blob Storage or other accessible storage.
Key Sections in the Notebook
Importing Libraries: The notebook starts by importing essential Python libraries such as pandas, numpy, matplotlib, and sklearn for data processing and model building.

#Data Ingestion: Data is ingested from Azure Blob Storage using a mount point configured with credentials. If you encounter issues with the mount point, ensure you have the correct storage account key and container access.

#Data Preprocessing:

The CSV file is read into a Spark DataFrame, and necessary columns are selected.
The Spark DataFrame is then converted into a Pandas DataFrame for further processing.
Handling of missing values and encoding of categorical variables using LabelEncoder.
Feature Engineering:

Selection of features relevant to churn prediction.
Feature scaling using StandardScaler to normalize the data for better model performance.
Modeling:

Logistic Regression: Baseline model to predict customer churn.
Random Forest: Ensemble method to capture non-linear relationships.
Support Vector Machine (SVM): Linear SVM for classification.
k-Nearest Neighbors (k-NN): Non-parametric method for classification.
Gradient Boosting: Another ensemble method focusing on minimizing error using decision trees.
Model Evaluation:

Evaluation of each model using metrics like confusion matrix, classification report, and accuracy score.
Feature importance plot for Random Forest to understand the contribution of each feature.
Visualization:

Plot of feature importance based on Random Forest to highlight the most influential features.
Getting Started
Prerequisites:

Databricks Workspace with access to Azure Blob Storage.
Required Python libraries installed (pandas, numpy, matplotlib, seaborn, sklearn).
Setting Up the Environment:

Make sure to configure the storage account and container with appropriate access keys in the Ingest Data section.
Upload the WA_Fn-UseC_-Telco-Customer-Churn.csv dataset to your storage container and mount it to /mnt/churn in the Databricks environment.
Running the Notebook:

Run the notebook cells sequentially to execute the entire workflow.
Modify the feature set and hyperparameters as needed based on your experimentation.
Results
The models provide varying levels of accuracy and insights into customer churn. The Random Forest and Gradient Boosting models typically show better performance for this dataset, as indicated by their higher accuracy scores and detailed feature importance visualization.
