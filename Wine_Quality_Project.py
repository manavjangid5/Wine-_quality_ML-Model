import pandas
import numpy
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,precision_score, recall_score, f1_score, accuracy_score

df = pandas.read_csv("winequality-red.csv",delimiter=';')
print(df.describe())

X = df[['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
Y = df['quality']

# Splitting the dataset into 70% training data and 30% testing data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=0)

print(Y_test.values)

############################ Linear Regression ###############################
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train.values,Y_train)

Y_pred = regr.predict(X_test.values)
print("------------------Linear Regression---------------")
y_pred1 = []
for val in Y_pred:
    y_pred1.append(round(val))

Y_pred1 = numpy.array(y_pred1)

print(Y_pred1)

#Creating confusion matrix
cm = confusion_matrix(Y_test, Y_pred1, labels=[0,1,2,3,4,5,6,7,8,9])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1,2,3,4,5,6,7,8,9])

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (Regression)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print('Accuracy: %.3f' % accuracy_score(Y_test, Y_pred1))

########################### KNN ###########################################

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
  
knn.fit(X_train.values, Y_train)

Y_pred_knn = knn.predict(X_test.values)
print("--------------KNN-------------")
print(Y_pred_knn)

#Creating confusion matrix
cm = confusion_matrix(Y_test, Y_pred_knn, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knn.classes_)

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (KNN)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print('Accuracy: %.3f' % accuracy_score(Y_test, Y_pred_knn))

########################### DT ###########################################

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
 
dtree.fit(X_train.values, Y_train)

Y_pred_dt = dtree.predict(X_test.values)

#Creating confusion matrix
cm = confusion_matrix(Y_test, Y_pred_dt, labels=dtree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=dtree.classes_)

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (DT)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("--------------DT-------------")
print(Y_pred_dt)

print('Accuracy: %.3f' % accuracy_score(Y_test, Y_pred_dt))


########################### Random Forest ###########################################

from sklearn.ensemble import RandomForestClassifier
rForest = RandomForestClassifier()
 
rForest.fit(X_train.values, Y_train)

Y_pred_rForest = rForest.predict(X_test.values)

#Creating confusion matrix
cm = confusion_matrix(Y_test, Y_pred_rForest, labels=rForest.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rForest.classes_)

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (Random Forest)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("--------------Random Forest-------------")
print(Y_pred_rForest)

print('Accuracy: %.3f' % accuracy_score(Y_test, Y_pred_rForest))



########################### Naive Bayes ###########################################

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

gnb=GaussianNB()
mnb=MultinomialNB()
gnb.fit(X_train.values, Y_train)
mnb.fit(X_train.values, Y_train)

Y_pred_gnb = gnb.predict(X_test.values)
Y_pred_mnb = mnb.predict(X_test.values)

#Creating confusion matrix
cm = confusion_matrix(Y_test, Y_pred_gnb, labels=gnb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=gnb.classes_)

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (GNB)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("--------------GNB-------------")
print(Y_pred_gnb)

print('Accuracy: %.3f' % accuracy_score(Y_test, Y_pred_gnb))


#Creating confusion matrix
cm = confusion_matrix(Y_test, Y_pred_mnb, labels=mnb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mnb.classes_)

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (MNB)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("--------------MNB-------------")
print(Y_pred_mnb)

print('Accuracy: %.3f' % accuracy_score(Y_test, Y_pred_mnb))


########################## SVM ###########################################

from sklearn import svm
rbf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
rbf.fit(X_train, Y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1.0)
poly.fit(X_train, Y_train)
svc = svm.SVC(kernel='linear', C=1.0)
svc.fit(X_train, Y_train)
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
svc_pred = svc.predict(X_test)

#Creating confusion matrix
cm = confusion_matrix(Y_test, poly_pred, labels=poly.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=poly.classes_)

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (SVM-Poly)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("--------------POLY-------------")
print(poly_pred)

print('Accuracy: %.3f' % accuracy_score(Y_test, poly_pred))


####################################RBF###########################################

#Creating confusion matrix
cm = confusion_matrix(Y_test, rbf_pred, labels=rbf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rbf.classes_)

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (SVM-RBF)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("--------------RBF-------------")
print(rbf_pred)

print('Accuracy: %.3f' % accuracy_score(Y_test, rbf_pred))


######################################### Linear ##############################

#Creating confusion matrix
cm = confusion_matrix(Y_test, svc_pred, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=svc.classes_)

#Plotting the confusion matrix
disp.plot()
plt.title('Confusion Matrix (SVM-Linear)')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("--------------Linear-------------")
print(svc_pred)

print('Accuracy: %.3f' % accuracy_score(Y_test, svc_pred))
