# Customer Churn Analysis
## Overview:
This repository provides an end-to-end machine learning workflow to predict customer churn in a telecommunications company using Databricks. The notebook includes data ingestion, preprocessing, feature engineering, model training, evaluation, and visualization of results. It demonstrates a comparison of multiple classification models including Logistic Regression, Random Forest, Support Vector Machines, k-Nearest Neighbors, and Gradient Boosting.

## Files and Structure:
notebook.ipynb: The main Databricks notebook where all the analysis and modeling are performed.
README.md: A file explaining the purpose and structure of the repository.
WA_Fn-UseC_-Telco-Customer-Churn.csv: The dataset used to train and test the churn model, held in Azure Blob Storage

## Key Sections in the Notebook

**Importing Libraries:** 

The notebook starts by importing essential Python libraries such as pandas, numpy, matplotlib, and sklearn for data processing and model building.

**Data Ingestion:** 

Data is ingested from Azure Blob Storage using a mount point configured with credentials. 

**Data Preprocessing:**

The CSV file is read into a Spark DataFrame, and necessary columns are selected.
The Spark DataFrame is then converted into a Pandas DataFrame for further processing.
Handling of missing values and encoding of categorical variables using LabelEncoder.

**Feature Engineering:**

Selection of features relevant to churn prediction.
Feature scaling using StandardScaler to normalize the data for better model performance.

**Modeling:**

***Logistic Regression:*** 

Baseline model to predict customer churn.

***Random Forest:***

Ensemble method to capture non-linear relationships.

***Support Vector Machine (SVM):***

Linear SVM for classification.

***k-Nearest Neighbors (k-NN):***

Non-parametric method for classification.

***Gradient Boosting:***

Another ensemble method focusing on minimizing error using decision trees.

**Model Evaluation:**

Evaluation of each model using metrics like confusion matrix, classification report, and accuracy score.
Feature importance plot for Random Forest to understand the contribution of each feature.

**Visualization:**

Plot of feature importance based on Random Forest to highlight the most influential features.


## Results

The models provide varying levels of accuracy and insights into customer churn. The Random Forest and Gradient Boosting models typically show better performance for this dataset, as indicated by their higher accuracy scores and detailed feature importance visualization.

## Future Enhancements

Since this was done in Databricks, a Spark DataFrame should be utilized instead of a Pandas DataFrame to take full advantage of distributed computing across nodes. However, given that this is a small dataset, using a Pandas DataFrame is acceptable. If the dataset were larger, leveraging a Spark DataFrame would be more appropriate.

It’s also important to consider that if the dataset is continuously updated upstream, it would be more efficient to read it directly from a table in the Hive metastore, rather than using `dbfs.utils` to virtually mount the CSV from blob storage. However, since the dataset is not expected to change frequently, using `dbfs.utils` is a suitable approach in this case.
