# Parkinson-s-Disease
!pip install opendatasets

import opendatasets as od

import pandas

od.download(

  "https://www.kaggle.com/competitions/amp-Parkinson’s-disease-progression-prediction/data")
  
import pandas as pd
import io
clinical_data= pd.read_csv('/content/amp-Parkinson’s-disease-progression-prediction/train_clinical_data.csv')
peptides_data=pd.read_csv('/content/amp-Parkinson’s-disease-progression-prediction/train_peptides.csv')
proteins_data=pd.read_csv('/content/amp-Parkinson’s-disease-progression-prediction/train_proteins.csv')
clinical_data.head()
clinical_data.columns
cd_data=[]
i1=0
j1=0
l1=-1
for i,j in clinical_data[["patient_id","visit_month"]].values.tolist():
  if(j==0 and i1!=i):
    cd_data.append(0)
    i1=i
    j1=j
  elif(i1!=i):
    cd_data.append(j)
    j1=j
    i1=i
  else:
    cd_data.append(j-j1)
    j1=j
len(cd_data)
clinical_data.shape
clinical_data["visit_diff"]=cd_data
clinical_data.head()
clinical_data['visit_diff'].plot(kind='hist')
proteins_data.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(proteins_data['UniProt'])
proteins_data['UniProt']=le.transform(proteins_data['UniProt'])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(proteins_data['NPX'].to_numpy().reshape(-1, 1))
proteins_data['NPX']=scaler.fit_transform(proteins_data['NPX'].to_numpy().reshape(-1, 1))
proteins_data.head()
proteins_data["UniProt"].nunique()
prot_count=proteins_data[["visit_id","visit_month"]].value_counts()
prot_count
prot_clinical_table=pd.merge(clinical_data,prot_count.to_frame(),how="left", on="visit_id")
prot_clinical_table.head()
prot_clinical_table.rename(columns = {0:'prot_count'}, inplace = True)
prot_clinical_table.isnull().sum()
peptides_data.head()
peptide_count=proteins_data[["visit_id","visit_month"]].value_counts()
peptide_count
final_table=pd.merge(prot_clinical_table,peptide_count.to_frame(),how="left", on="visit_id")
final_table.head()
#final_table= final_table.drop(final_table[final_table['visit_diff'] == 0].index)
final_table.rename(columns = {0:'pep_count'}, inplace = True)
final_table.head()
final_table["prot_count"].equals(final_table["pep_count"])
final_table["prot_count"]=(final_table['prot_count'] > 0).astype(int)
final_table.head()
final_table["pep_count"].fillna(0, inplace = True)
final_table.head()
final_table=final_table.dropna(subset=["updrs_4"])
final_table.isnull().sum()
final_table.shape
final_table["upd23b_clinical_state_on_medication"].unique
final_table=pd.DataFrame(final_table)
final_table.head()
final_table.rename(columns = {"upd23b_clinical_state_on_medication":'medication'}, inplace = True)
final_table.head()
final_table.isnull().sum()
final_table["medication"].fillna("Off", inplace = True)
final_table=final_table.dropna(subset=["updrs_2","updrs_3"])
final_table.isnull().sum()
final_table=pd.DataFrame(final_table)
final_table.columns
final_table = final_table.melt(id_vars=['visit_id', 'patient_id', 'visit_month',  'medication', 'visit_diff', 'prot_count',
       'pep_count'],
                 var_name='updrs', value_name='rating')
final_table.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(final_table['updrs'])
final_table['updrs']=le.transform(final_table['updrs'])
from sklearn import preprocessing
le1 = preprocessing.LabelEncoder()
le1.fit(final_table['medication'])
final_table['medication']=le1.transform(final_table['medication'])
final_table.head,final_table.shape
clinical_data['updrs_1'].plot(kind='kde')
clinical_data['updrs_2'].plot(kind='kde')
clinical_data['updrs_3'].plot(kind='kde')
clinical_data['updrs_4'].plot(kind='kde')
r=[]
t1,t2,t3,t4=7,7,15,5
for i,j in final_table[["updrs","rating"]].values.tolist():
  if i==0:
    if j>=t1:
      r.append(1)
    else:
      r.append(0)
  elif i==1:
    if j>=t2:
      r.append(1)
    else:
      r.append(0)
  elif i==2:
    if j>=t3:
      r.append(1)
    else:
      r.append(0)
  elif i==3:
    if j>=t4:
      r.append(1)
    else:
      r.append(0)
len(r),final_table.shape
final_table["rating"]=r
final_table.head()
final_table.rename(columns = {"prot_count":'prot_test'}, inplace = True)
final_table.dtypes
final_table.head()
import pandas as pd
pd.plotting.radviz(final_table[['visit_month', 'medication', 'visit_diff','prot_test', 'pep_count', 'updrs','rating']],'rating')
#for bar chart
name=['AUC_scores','Accuracy','F1 scores','Recall','Precision']
name_model=['Gradient Boosting','Neural Network','Random Forest','Graneau']
auc_s=[]
aucc=[]
f1_s=[]
recal=[]
prec=[]
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib
import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,mean_absolute_error,mean_squared_error
import joblib
import time
X = final_table[['visit_month', 'medication', 'visit_diff','prot_test', 'pep_count', 'updrs']]
y = final_table['rating']
models = {
    'Gradient Boosting': GradientBoostingClassifier(
        learning_rate=0.01,
        n_estimators=100,
        max_depth=3,
        subsample=0.8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        loss='log_loss',
    )
}
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
results = {}
results1 = {}
sum_of_accuracies = 0.0

for model_name, model in models.items():
    auc_scores = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    mae_scores=[]
    mse_scores=[]
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(f"Running model: {model_name}, Fold: {fold_idx + 1}")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training time for Fold {fold_idx + 1}: {elapsed_time / 60:.2f} minutes")
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mae=mean_absolute_error(y_test, y_pred)
        mse=mean_squared_error(y_test, y_pred)
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        mae_scores.append(mae)
        mse_scores.append(mse)
    avg_auc = np.mean(auc_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_mae=np.mean(mae_scores)
    avg_mse=np.mean(mse_scores)
    results[model_name] = {
        'Average AUC': avg_auc,
        'Average Accuracy': avg_accuracy,
        'Average F1 Score': avg_f1,
        'Average Precision': avg_precision,
        'Average Recall': avg_recall,
        'Average mae':avg_mae,
        'Average mse':avg_mse
    }
    results1[model_name] = {
        'max AUC': max(auc_scores),
        'max Accuracy': max(accuracy_scores),
        'max F1 Score': max(f1_scores),
        'max Precision': max(precision_scores),
        'max Recall': max(recall_scores),
        'max mae':max(mae_scores),
        'max mse':max(mse_scores)
    }
    sum_of_accuracies += avg_accuracy
    print(results)
    print(results1)
best_model_name = max(results, key=lambda k: results[k]['Average AUC'])
best_model_results = results[best_model_name]
print(f"The best model is {best_model_name} with the following average results over {n_splits} folds:")
print(f"Average AUC: {best_model_results['Average AUC']}")
print(f"Average Accuracy: {best_model_results['Average Accuracy']}")
print(f"Average F1 Score: {best_model_results['Average F1 Score']}")
print(f"Average Precision: {best_model_results['Average Precision']}")
print(f"Average Recall: {best_model_results['Average Recall']}")
print(f"Average mae: {best_model_results['Average mae']}")
print(f"Average mse: {best_model_results['Average mse']}")
auc_s.append(best_model_results['Average AUC'])
aucc.append(best_model_results['Average Accuracy'])
f1_s.append(best_model_results['Average F1 Score'])
recal.append(best_model_results['Average Recall'])
prec.append(best_model_results['Average Precision'])
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib
import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,mean_absolute_error,mean_squared_error
import joblib
import time
threshold = 5.0
X = final_table[['visit_month', 'medication', 'visit_diff','prot_test', 'pep_count', 'updrs']]
y = final_table['rating']
models = {
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(8,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
    ),
}
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
results = {}
sum_of_accuracies = 0.0
for model_name, model in models.items():
    auc_scores = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    mae_scores=[]
    mse_scores=[]
    n=0
    for fold_idx in range(5):
        n+=1
        print(f"Running model: {model_name}, Fold: {fold_idx + 1}")
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training time for Fold {fold_idx + 1}: {elapsed_time / 60:.2f} minutes")
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mae=mean_absolute_error(y_test, y_pred)
        mse=mean_squared_error(y_test, y_pred)
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        mae_scores.append(mae)
        mse_scores.append(mse)
    avg_auc = np.mean(auc_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_mae=np.mean(mae_scores)
    avg_mse=np.mean(mse_scores)
    results[model_name] = {
        'Average AUC': avg_auc,
        'Average Accuracy': avg_accuracy,
        'Average F1 Score': avg_f1,
        'Average Precision': avg_precision,
        'Average Recall': avg_recall,
        'Average mae':avg_mae,
        'Average mse':avg_mse
    }
    results1[model_name] = {
        'max AUC': max(auc_scores),
        'max Accuracy': max(accuracy_scores),
        'max F1 Score': max(f1_scores),
        'max Precision': max(precision_scores),
        'max Recall': max(recall_scores),
        'max mae':max(mae_scores),
        'max mse':max(mse_scores)
    }
    sum_of_accuracies += avg_accuracy
    print(results)
    print(results1)
best_model_name = max(results, key=lambda k: results[k]['Average AUC'])
best_model_results = results[best_model_name]
print(f"The best model is {best_model_name} with the following average results over {n_splits} folds:")
print(f"Average AUC: {best_model_results['Average AUC']}")
print(f"Average Accuracy: {best_model_results['Average Accuracy']}")
print(f"Average F1 Score: {best_model_results['Average F1 Score']}")
print(f"Average Precision: {best_model_results['Average Precision']}")
print(f"Average Recall: {best_model_results['Average Recall']}")
print(f"Average mae: {best_model_results['Average mae']}")
print(f"Average mse: {best_model_results['Average mse']}")
auc_s.append(best_model_results['Average AUC'])
aucc.append(best_model_results['Average Accuracy'])
f1_s.append(best_model_results['Average F1 Score'])
recal.append(best_model_results['Average Recall'])
prec.append(best_model_results['Average Precision'])
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib
import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,mean_absolute_error,mean_squared_error
import joblib
import time
X = final_table[['visit_month', 'medication', 'visit_diff','prot_test', 'pep_count', 'updrs']]
y = final_table['rating']
models = {
    'Random Forest': RandomForestClassifier(n_estimators= 100,criterion= 'gini',random_state= 42),
    'Stochastic Gradient Descent': SGDClassifier(
        loss='hinge',
        alpha=0.00001,
        learning_rate='constant',
        eta0=0.01,
    )
}
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
results = {}
results1 = {}
sum_of_accuracies = 0.0
for model_name, model in models.items():
    auc_scores = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    mae_scores=[]
    mse_scores=[]
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(f"Running model: {model_name}, Fold: {fold_idx + 1}")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training time for Fold {fold_idx + 1}: {elapsed_time / 60:.2f} minutes")
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mae=mean_absolute_error(y_test, y_pred)
        mse=mean_squared_error(y_test, y_pred)
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        mae_scores.append(mae)
        mse_scores.append(mse)
    avg_auc = np.mean(auc_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_mae=np.mean(mae_scores)
    avg_mse=np.mean(mse_scores)
    results[model_name] = {
        'Average AUC': avg_auc,
        'Average Accuracy': avg_accuracy,
        'Average F1 Score': avg_f1,
        'Average Precision': avg_precision,
        'Average Recall': avg_recall,
        'Average mae':avg_mae,
        'Average mse':avg_mse
    }
    results1[model_name] = {
        'max AUC': max(auc_scores),
        'max Accuracy': max(accuracy_scores),
        'max F1 Score': max(f1_scores),
        'max Precision': max(precision_scores),
        'max Recall': max(recall_scores),
        'max mae':max(mae_scores),
        'max mse':max(mse_scores)
    }
    sum_of_accuracies += avg_accuracy
    print(results)
    print(results1)
best_model_name = max(results, key=lambda k: results[k]['Average AUC'])
best_model_results = results[best_model_name]
print(f"The best model is {best_model_name} with the following average results over {n_splits} folds:")
print(f"Average AUC: {best_model_results['Average AUC']}")
print(f"Average Accuracy: {best_model_results['Average Accuracy']}")
print(f"Average F1 Score: {best_model_results['Average F1 Score']}")
print(f"Average Precision: {best_model_results['Average Precision']}")
print(f"Average Recall: {best_model_results['Average Recall']}")
print(f"Average mae: {best_model_results['Average mae']}")
print(f"Average mse: {best_model_results['Average mse']}")
auc_s.append(best_model_results['Average AUC'])
aucc.append(best_model_results['Average Accuracy'])
f1_s.append(best_model_results['Average F1 Score'])
recal.append(best_model_results['Average Recall'])
prec.append(best_model_results['Average Precision'])
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib
import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,mean_absolute_error,mean_squared_error
import joblib
import time
X = final_table[['visit_month', 'medication', 'visit_diff','prot_test', 'pep_count', 'updrs']]
y = final_table['rating']
models = {
    'Gradient Boosting': GradientBoostingClassifier(
        learning_rate=0.01,
        n_estimators=100,
        max_depth=3,
        subsample=0.8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        loss='log_loss',
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(8,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
    ),
}
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
results = {}
results1 = {}
sum_of_accuracies = 0.0
for model_name, model in models.items():
    auc_scores = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    mae_scores=[]
    mse_scores=[]
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(f"Running model: {model_name}, Fold: {fold_idx + 1}")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training time for Fold {fold_idx + 1}: {elapsed_time / 60:.2f} minutes")
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mae=mean_absolute_error(y_test, y_pred)
        mse=mean_squared_error(y_test, y_pred)
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        mae_scores.append(mae)
        mse_scores.append(mse)
    avg_auc = np.mean(auc_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_mae=np.mean(mae_scores)
    avg_mse=np.mean(mse_scores)
    results[model_name] = {
        'Average AUC': avg_auc,
        'Average Accuracy': avg_accuracy,
        'Average F1 Score': avg_f1,
        'Average Precision': avg_precision,
        'Average Recall': avg_recall,
        'Average mae':avg_mae,
        'Average mse':avg_mse
    }
    results1[model_name] = {
        'max AUC': max(auc_scores),
        'max Accuracy': max(accuracy_scores),
        'max F1 Score': max(f1_scores),
        'max Precision': max(precision_scores),
        'max Recall': max(recall_scores),
        'max mae':max(mae_scores),
        'max mse':max(mse_scores)
    }
    sum_of_accuracies += avg_accuracy
    print(results)
    print(results1)
best_model_name = max(results, key=lambda k: results[k]['Average AUC'])
best_model_results = results[best_model_name]
print(f"The best model is {best_model_name} with the following average results over {n_splits} folds:")
print(f"Average AUC: {best_model_results['Average AUC']}")
print(f"Average Accuracy: {best_model_results['Average Accuracy']}")
print(f"Average F1 Score: {best_model_results['Average F1 Score']}")
print(f"Average Precision: {best_model_results['Average Precision']}")
print(f"Average Recall: {best_model_results['Average Recall']}")
print(f"Average mae: {best_model_results['Average mae']}")
print(f"Average mse: {best_model_results['Average mse']}")
auc_s.append(best_model_results['Average AUC'])
aucc.append(best_model_results['Average Accuracy'])
f1_s.append(best_model_results['Average F1 Score'])
recal.append(best_model_results['Average Recall'])
prec.append(best_model_results['Average Precision'])
import matplotlib.pyplot as plt
i=-0.3
X_axis = np.arange(len(name_model))

plt.bar(X_axis+i,auc_s,0.1,label=name[0])
i+=0.1
plt.bar(X_axis+i,aucc,0.1,label=name[1])
i+=0.1
plt.bar(X_axis+i,f1_s,0.1,label=name[2])
i+=0.1
plt.bar(X_axis+i,recal,0.1,label=name[3])
i+=0.1
plt.bar(X_axis+i,prec,0.1,label=name[4])
i+=0.1
plt.xticks(X_axis, name_model)
plt.xlabel("Models")
plt.ylabel("Scores")
plt.title("Performance of models")
plt.legend()
plt.show()
def predict_values(X,models):
  model_value=[1,0]
  l=[]
  for model_name, model in models.items():
      l.append(model.predict(X))
  n=len(l[0].tolist())
  y=[]
  for j in range(n):
    d={}
    for i in l:
        if i[j] in d:
          d[i[j]]+=1
        else:
          d[i[j]]=1
    y.append(max(d, key = d.get) )
  return y
l=predict_values(X,models)
len(l)
