import pandas as pd
data=pd.read_csv("C:\\Users\\91936\\Downloads\\archive (1)\\creditcard.csv")

#Find Shape:
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])


#Check null values:
data.isnull().sum()

##Handling Duplicates:

#StandardScalar:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))
data=data.drop(["Time"],axis=1)
data.shape

#Dropping Duplicates
data=data.drop_duplicates()
data.shape

##Not Handling Duplicates:
data['Class'].value_counts()

#Visualize:
import seaborn as sns
sns.countplot(data['Class'])

#Store Feature Matrix In X And Target In vector y:
X=data.drop('Class',axis=1)
y=data['Class']


#Splitting the Dataset into Training set and Testing set:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

#Handling Imbalanced Dataset:
     #undersampling
     #oversampling

#undersampling
normal=data[data['Class']==0]
fraud=data[data['Class']==1]

normal_sample=normal.sample(n=473)
new_data=pd.concat([normal_sample,fraud],ignore_index=True)
new_data['Class'].value_counts()

X=new_data.drop('Class',axis=1)
y=new_data['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
#Logistic Regression:

from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)

y_pred1=log.predict(X_test)

#Decision Tree Classifier:
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred2=dt.predict(X_test)

#Random Forest Classifier:
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred3=rf.predict(X_test)

#final data:
from sklearn.metrics import accuracy_score
final_data=pd.DataFrame({'Models':['LR','DT','RF'],'ACC':[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})
print("Final data:", final_data)

#Oversampling:
X=data.drop('Class',axis=1)
y=data['Class']

from imblearn.over_sampling import SMOTE
X_res,y_res= SMOTE().fit_resample(X,y)
y_res.value_counts

X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.20,random_state=42)

#Logistic Regression:
log=LogisticRegression()
log.fit(X_train,y_train)
y_pred1=log.predict(X_test)

#Decision Tree Classifier:
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred2=log.predict(X_test)

#Random Forest Classifier:
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred3=log.predict(X_test)

final_data1=pd.DataFrame({'Models':['LR','DT','RF'],'ACC':[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})
print("Accuracy of three models:",final_data1)

#Save The Model:
rf1=RandomForestClassifier()
rf1.fit(X_res,y_res)

import joblib
joblib.dump(rf1,"credit_card_model")
model=joblib.load("credit_card_model")

pred=model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

if pred == 0:
  print("Normal Transaction")
else:
  print("Fraudulent Transaction")
