import pandas as pd
heart=pd.read_csv('C://Users//ChandanA//Desktop//class//heart.csv')
print(heart.shape)
#DATA CLEANING
heart=heart.dropna() #drop missing values
#REPLACE MISSING VALUES
#heart=heart.ffill(inplace=True)- FORWARD FILL
#heart=heart.bfill(inplace=True) - BACKWARD FILL
#heart=heart.interpolate()-
#data integration
heart.drop_duplicates(keep=False,inplace=True) #drops duplicate rows
print(heart.shape)
att_names=['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']
X=heart[att_names]
Y=heart['output']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#data transformation
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#logistic regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
print('Accuracy of logistic Regression on trained dataset :{:.2f}'
      .format(logreg.score(X_train,Y_train)))
print('Accuracy of logistic Regression on test dataset :{:.2f}'
      .format(logreg.score(X_test,Y_test)))
#decision tree
from sklearn.tree import DecisionTreeClassifier
res=DecisionTreeClassifier()
res.fit(X_train,Y_train)
print('Accuracy of Decision Tree on trained dataset :{:.2f}'
      .format(res.score(X_train,Y_train)))
print('Accuracy of Decision Tree on test dataset :{:.2f}'
      .format(res.score(X_test,Y_test)))
#knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
print('Accuracy of KNN on trained dataset :{:.2f}'
      .format(knn.score(X_train,Y_train)))
print('Accuracy of KNN on test dataset :{:.2f}'
      .format(knn.score(X_test,Y_test)))
#naivebayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,Y_train)
print('Accuracy of naive bayes on trained dataset :{:.2f}'
      .format(gnb.score(X_train,Y_train)))
print('Accuracy of naive bayes on test dataset :{:.2f}'
      .format(gnb.score(X_test,Y_test)))

#svm
from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,Y_train)
print('Accuracy of SVM on trained dataset :{:.2f}'
      .format(svm.score(X_train,Y_train)))
print('Accuracy of SVM on test dataset :{:.2f}'
      .format(svm.score(X_test,Y_test)))

gamma='auto'
#confusion matrix for knn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred=knn.predict(X_test)
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred))
#INPUT-1
inp=[[56,0,0,200,288,1,0,133,1,4,0,2,3]]
h=knn.predict(inp)
print('output for given input',h)
inp=[[56,1,0,130,283,1,0,103,1,1.6,0,0,3]]
h=knn.predict(inp)
print('output for given input',h)
#ACTUAL AND PREDICTED
#knn.fit(X_test,Y_test)
pred=knn.predict(X_test)
df=pd.DataFrame({'Actual ':Y_test,'Predicted':pred})
print(df)







