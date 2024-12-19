
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, make_scorer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from lifelines import CoxPHFitter
from sksurv.svm import FastKernelSurvivalSVM

import matplotlib.pyplot as plt
import seaborn as sns

"""### 2. Loading and Preparing the Datasets Separately"""

# Load the datasets
df_patient = pd.read_csv("cleaned patient data.csv")
df_mrna = pd.read_csv("patient mrna data.csv")

# Preprocess df_patient
df_patient.replace('NA', pd.NA, inplace=True)
numeric_columns = [
    'age_at_diagnosis', 'tumor_size', 'er_status', 'pgr_status',
    'overall_survival_days', 'overall_survival_years',
    'overall_survival_event', 'relapse_free_interval_days', 'relapse_free_interval_years',
    'relapse_free_interval_event', 'esr1_expression:_log2(tpm+0.1)', 'esr2_expression:_log2(tpm+0.1)'
]
for col in numeric_columns:
    df_patient[col] = pd.to_numeric(df_patient[col], errors='coerce')

# Identify numeric columns for scaling
columns_to_normalize_patient = [
    col for col in df_patient.columns if col not in ['overall_survival_days', 'overall_survival_event']
    and pd.api.types.is_numeric_dtype(df_patient[col])
]

# Initialize the scaler and apply it to the numeric columns only
ss_patient = MinMaxScaler()
df_norm_patient = pd.DataFrame(ss_patient.fit_transform(df_patient[columns_to_normalize_patient]), columns=columns_to_normalize_patient)

# Add the survival columns back to the normalized dataframe
df_norm_patient['overall_survival_days'] = df_patient['overall_survival_days']
df_norm_patient['overall_survival_event'] = df_patient['overall_survival_event']

# Fill missing values with the mean of each column in df_norm_patient
df_norm_patient.fillna(df_norm_patient.mean(), inplace=True)

# Preprocess df_mrna
# Identify numeric columns in df_mrna
numeric_columns_mrna = df_mrna.select_dtypes(include=[np.number]).columns.tolist()

# Initialize the scaler and scale only numeric columns
ss_mrna = MinMaxScaler()
df_norm_mrna = pd.DataFrame(ss_mrna.fit_transform(df_mrna[numeric_columns_mrna]), columns=numeric_columns_mrna)

# Fill missing values with the mean of each column in df_norm_mrna
df_norm_mrna.fillna(df_norm_mrna.mean(), inplace=True)

"""### 3. Hierarchical Clustering on Each Dataset"""

# Hierarchical Clustering on df_patient
X_patient = df_norm_patient.drop(['overall_survival_days', 'overall_survival_event'], axis=1)
clustering_patient = AgglomerativeClustering(n_clusters=5)
labels_patient = clustering_patient.fit_predict(X_patient)

# Evaluation metrics for df_patient
silhouette_avg_patient = silhouette_score(X_patient, labels_patient)
davies_bouldin_avg_patient = davies_bouldin_score(X_patient, labels_patient)
calinski_harabasz_avg_patient = calinski_harabasz_score(X_patient, labels_patient)

# Plot dendrogram for df_patient
Z_patient = linkage(X_patient, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(Z_patient)
plt.title('Hierarchical Clustering Dendrogram (df_patient)')
plt.show()

# Hierarchical Clustering on df_mrna
clustering_mrna = AgglomerativeClustering(n_clusters=5)
labels_mrna = clustering_mrna.fit_predict(df_norm_mrna)

# Evaluation metrics for df_mrna
silhouette_avg_mrna = silhouette_score(df_norm_mrna, labels_mrna)
davies_bouldin_avg_mrna = davies_bouldin_score(df_norm_mrna, labels_mrna)
calinski_harabasz_avg_mrna = calinski_harabasz_score(df_norm_mrna, labels_mrna)

# Plot dendrogram for df_mrna
Z_mrna = linkage(df_norm_mrna, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(Z_mrna)
plt.title('Hierarchical Clustering Dendrogram (df_mrna)')
plt.show()

"""### 4. Feature Importance with Random Survival Forest on df_patient"""

# Structured array for survival data in df_patient
y_patient = np.array([(event, time) for event, time in zip(df_norm_patient['overall_survival_event'], df_norm_patient['overall_survival_days'])],
                     dtype=[('event', 'bool'), ('time', 'float')])
X_patient_features = df_norm_patient.drop(['overall_survival_days', 'overall_survival_event'], axis=1)

# Random Survival Forest on df_patient with hyperparameter tuning
param_grid_rsf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 20, 30],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [5, 10, 15],
    'max_features': ['sqrt'],
    'bootstrap': [True, False],
}
rsf_patient = RandomSurvivalForest(random_state=42)
rsf_random_patient = RandomizedSearchCV(estimator=rsf_patient, param_distributions=param_grid_rsf, n_iter=10, cv=3, scoring=make_scorer(concordance_index_censored, greater_is_better=True), n_jobs=2)
rsf_random_patient.fit(X_patient_features, y_patient)

# Evaluate model on df_patient
best_rsf_patient = rsf_random_patient.best_estimator_
y_pred_patient = best_rsf_patient.predict(X_patient_features)
c_index_patient = concordance_index_censored(y_patient['event'], y_patient['time'], y_pred_patient)

# Feature importance for df_patient
result_patient = permutation_importance(best_rsf_patient, X_patient_features, y_patient, n_repeats=5, random_state=42, n_jobs=1)
feature_importances_patient = pd.Series(result_patient.importances_mean, index=X_patient_features.columns).sort_values(ascending=False)
feature_importances_patient.to_csv("feature_importances_patient.csv")

"""### 5. Survival Analysis with SVM on df_patient"""

# SVM for survival analysis on df_patient
param_grid_svm = {
    'alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization parameter
    'kernel': ['linear', 'rbf'],       # Kernel type
    'gamma': [0.1, 1.0, 10.0]          # Gamma parameter as a positive float
}
svm_patient = FastKernelSurvivalSVM()
grid_search_patient = GridSearchCV(estimator=svm_patient, param_grid=param_grid_svm, cv=3, scoring=make_scorer(concordance_index_censored, greater_is_better=True))
grid_search_patient.fit(X_patient_features, y_patient)

# Best SVM model on df_patient
best_svm_patient = grid_search_patient.best_estimator_

# Predict on the same dataset (since there's no merging, training/testing is based on df_patient only)
risk_scores_patient = best_svm_patient.predict(X_patient_features)

# Evaluate SVM model using Concordance Index on df_patient
c_index_svm_patient = concordance_index_censored(y_patient['event'], y_patient['time'], risk_scores_patient)

print(f'Best Parameters (SVM): {grid_search_patient.best_params_}')
print(f'Concordance Index (SVM on df_patient): {c_index_svm_patient[0]:.4f}')

# Plot risk scores distribution for df_patient
plt.hist(risk_scores_patient, bins=20)
plt.title("Distribution of Risk Scores (SVM on df_patient)")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.show()
