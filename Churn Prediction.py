# Databricks notebook source
# DBTITLE 1,Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# COMMAND ----------

# DBTITLE 1,Ingest Data
storage_account_name = "dataworkspace123"
container_name = "churn"
storage_account_access_key = "4H2A7dVHiVwhYiQLSpD/eRxqdmfv6WeS6fJBd4NCDgS1SAAy1++gf70N6aEaTPHz97Gquc21d/3W+AStoZU2JA=="

try:
    dbutils.fs.mount(
        source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
        mount_point = f"/mnt/{container_name}",
        extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key}
    )

except:
    print("Mount point already exists")




# COMMAND ----------

# DBTITLE 1,Verifying Datatypes
df = spark.read.csv("/mnt/churn/WA_Fn-UseC_-Telco-Customer-Churn.csv", header=True, inferSchema=True)



# COMMAND ----------

# DBTITLE 1,Filtering Columns
df=df.select("customerID", "gender", "tenure", "PhoneService", "MultipleLines", "InternetService", "Contract", "TechSupport", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn")

df.show()

# COMMAND ----------

# DBTITLE 1,Converting to Pandas
pd_df = df.toPandas()
print(pd_df.isnull().sum())
print(pd_df.duplicated().sum())

#converted cause duplicated function is present in pandas only #

# COMMAND ----------

# DBTITLE 1,Converting datatypes
columns_to_encode = ['gender', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']

for col in columns_to_encode:
    pd_df[col] = LabelEncoder().fit_transform(pd_df[col])

pd_df['TotalCharges'] = pd.to_numeric(pd_df['TotalCharges'], errors='coerce')

pd_df['TotalCharges'] = pd_df['TotalCharges'].fillna(0)

pd_df.dtypes


# COMMAND ----------

# DBTITLE 1,Initializing Features and Test/Train data
features = ['gender', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

x = pd_df[features]
y = pd_df['Churn']

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

# COMMAND ----------

# DBTITLE 1,Feature Scaling
scaler = StandardScaler()

x_tr = scaler.fit_transform(x_tr)

x_te = scaler.transform(x_te)

# COMMAND ----------

# DBTITLE 1,Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_tr, y_tr)

y_p_lr=log_reg.predict(x_te)

conf_matrix_lr=confusion_matrix(y_te, y_p_lr)
class_report_lr=classification_report(y_te, y_p_lr)
accuracy_lr=accuracy_score(y_te, y_p_lr)

print(conf_matrix_lr)
print(class_report_lr)
print(accuracy_lr)


# COMMAND ----------

# DBTITLE 1,RandomForest
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(x_tr, y_tr)
y_p=model.predict(x_te)
conf_matrix=confusion_matrix(y_te, y_p)
class_report=classification_report(y_te, y_p)
accuracy=accuracy_score(y_te, y_p)

print(conf_matrix)
print(class_report)
print(accuracy)



# COMMAND ----------

# DBTITLE 1,Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)
names = [features[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(x.shape[1]), importances[indices])
plt.yticks(range(x.shape[1]), names)
plt.show()

# COMMAND ----------

# DBTITLE 1,Support Vector Machines
from sklearn.svm import SVC
sv_model = SVC(kernel='linear', random_state=42)
sv_model.fit(x_tr, y_tr)

y_p_svm = sv_model.predict(x_te)

conf_matrix_svm=confusion_matrix(y_te, y_p_svm)
class_report_svm=classification_report(y_te, y_p_svm)
accuracy_svm=accuracy_score(y_te, y_p_svm)

print(conf_matrix_svm)
print(class_report_svm)
print(accuracy_svm)

# COMMAND ----------

# DBTITLE 1,Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_tr, y_tr)

y_p_knn = knn_model.predict(x_te)

conf_matrix_knn=confusion_matrix(y_te, y_p_knn)
class_report_knn=classification_report(y_te, y_p_knn)
accuracy_knn=accuracy_score(y_te, y_p_knn)

print(conf_matrix_knn)
print(class_report_knn)
print(accuracy_knn)

# COMMAND ----------

# DBTITLE 1,Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gbm_model=GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(x_tr, y_tr)

y_p_gbm = gbm_model.predict(x_te)

conf_matrix_gbm=confusion_matrix(y_te, y_p_gbm)
class_report_gbm=classification_report(y_te, y_p_gbm)
accuracy_gbm=accuracy_score(y_te, y_p_gbm)

print(conf_matrix_gbm)
print(class_report_gbm)
print(accuracy_gbm)
