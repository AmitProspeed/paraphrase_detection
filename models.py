from feature_engineering import train_X, dev_X, test_X, train_y, dev_y, test_y
from sklearn.linear_model import LogisticRegression


#Logistic Regression

logreg = LogisticRegression(C=0.0021, solver='liblinear', random_state=42)
logreg.fit(train_X,train_y)
print("Logistic Regression training accuracy: {}".format(logreg.score(train_X, train_y)))
print("Logistic Regression dev accuracy: {}".format(logreg.score(dev_X,dev_y)))
print("Logistic Regression test accuracy: {}".format(logreg.score(test_X,test_y)))
print("F1:",metrics.f1_score(test_y, lr_pred))


#SVM

svclassifier = SVC(kernel='rbf', gamma=0.0001, C=5)
svclassifier.fit(train_X, train_y)
y_pred = svclassifier.predict(test_X)
print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))


#XG Boost

xgb1 = GradientBoostingClassifier(loss='deviance', #deviance
                                  learning_rate = 0.01, #0.1
                                  n_estimators= 250, #100
                                  subsample = 1, #1
                                  #min_samples_split = 0.2, #2
                                  min_samples_leaf = 1, #1
                                  min_weight_fraction_leaf = 0, #0
                                  max_depth = 9,
                                  min_impurity_decrease = 0, #0
                                  #min_impurity_split = 0,  #1e-7
                                  max_features = 'log2', #none
                                  max_leaf_nodes = None, #none
                                  validation_fraction = 0.1, #0.1
                                  n_iter_no_change = None, #none
                                  tol = 1e-4 #1e-4
                                  #ccp_alpha = 0 #0
                                 )
xgb1.fit(train_X, train_y)
y_pred_xgb1 = xgb1.predict(test_X)
print("Accuracy:",metrics.accuracy_score(test_y, y_pred_xgb1))
print("F1:",metrics.f1_score(test_y, y_pred_xgb1))



#Random Forest

rfclass1 = RandomForestClassifier(n_estimators = 4500, 
                                 max_depth=10, 
                                 random_state=0,
                                 #min_samples_split = 0.1,
                                 min_samples_leaf=3,
                                 max_features = 'log2',
                                 verbose =3)
rfclass1.fit(train_X, train_y)
y_pred_rf1 = rfclass1.predict(test_X)
print("Accuracy:",metrics.accuracy_score(test_y, y_pred_rf1))
print("F1:",metrics.f1_score(test_y, y_pred_rf1))


#Adaboost

adab1 = AdaBoostClassifier(n_estimators = 1000, 
                          learning_rate=0.0001,
                           random_state = 0)
                           #algorithm = 'SAMME')# verbose =3)
adab1.fit(train_X, train_y)
y_pred_ada1 = adab1.predict(test_X)
print("Accuracy:",metrics.accuracy_score(test_y, y_pred_ada1))
print("F1:",metrics.f1_score(test_y, y_pred_ada1))


#Decsiion Tree

dtree1 = DecisionTreeClassifier(criterion = 'entropy',
                            #min_samples_leaf = 0.3,
                            #splitter = 'best',
                            max_depth = 2,
                           random_state = 0)
                           #max_features = 'log2')
                           
dtree1.fit(train_X, train_y)
y_dtree = dtree1.predict(test_X)
print("Accuracy:",metrics.accuracy_score(test_y, y_dtree))
print("F1:",metrics.f1_score(test_y, y_dtree))


#Naive Bayes

from sklearn.naive_bayes import BernoulliNB
bnb1 = BernoulliNB()
bnb1.fit(train_X,train_y)
y_bnb1 = bnb1.predict(test_X)
print("Accuracy:",metrics.accuracy_score(test_y, y_bnb1))
print("F1:",metrics.f1_score(test_y, y_bnb1))