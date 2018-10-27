""" * Gender prediction using Twitter data *

This code builds the data model with the data present in twitter_data_model_duilding.csv

Three algorithms are applied here:
1.Logistic Regression( regularised)
2.Support Vector Machine
3.RandomForest

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import statsmodels.api as sm
import scipy.stats as ss
from sklearn.model_selection import GridSearchCV

"""Importing the Dataset for Model Building """
twitter_data=pd.read_csv(r'twitter_data_model_building.csv',sep=",",index_col='Unnamed: 0')

""" Performace evaluation function for binary classification models """
def PerformanceEvaluationMetrics(y_test,y_pred):
    #print("_________________________PERFORMACE EVALUATION_______________________________________")
    confusionMatrix=metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confusionMatrix)
    print("\n\n")
    plt.clf()
    plt.imshow(confusionMatrix, cmap=plt.cm.Wistia)
    classNames = ['Female','Male']
    plt.title('Male or Female Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(confusionMatrix[i][j]))
    plt.show()
    print("\n")
    
    TP = confusionMatrix[0,0] # true positive 
    TN = confusionMatrix[1,1] # true negatives
    FP = confusionMatrix[0,1] # false positives
    FN = confusionMatrix[1,0] # false negatives

    # accuracy
    print("accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))
    # precision
    print("precision:", round(metrics.precision_score(y_test, y_pred),2))
    # recall/sensitivity
    print("recall:", round(metrics.recall_score(y_test, y_pred),2))  
    # Sensitivity
    print("sensitivity:",round(TP / float(TP+FN),2))
    # Specificity
    print("specificity:",round(TN / float(TN+FP),2))
    #AUC
    auc_score = metrics.roc_auc_score( y_test, y_pred )
    print("AUC:", round(auc_score,3))
    
    #ROC
    fpr, tpr, thresholds = metrics.roc_curve( y_test, y_pred,
                                                  drop_intermediate = False ) 
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

""" Using Recursive feature elimination to plot the graphs Accuracy Vs number of features """
def RecursiveElimination(model,X_train,y_train,X_test,y_test):
    no_of_feat,acc=[],[]
    for i in reversed((list(range(1,len(x.columns)+1)))):
        print(i)
        rfe = RFE(model,i)
        rfe = rfe.fit(X_train,y_train)
        score=round(rfe.score(X_test,y_test),3)
        print(score)
        no_of_feat.append(i)
        acc.append( score)
        plt.plot(no_of_feat, acc)
    plt.title("No of features Vs Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('No of Features')
    plt.show()
    return   no_of_feat,acc

""" Twitter Data Scaling and Diving data into Test and Train """
x = twitter_data.drop('gender',axis=1)
y = twitter_data.gender
colnames=x.columns 
sc = StandardScaler()  
X  = sc.fit_transform(x)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


############################### Model Building ###################################### 
#Logistic regression
# Make the instance of the model

logisticRegrModel = LogisticRegression(C = 1.0,random_state = 2018)
logisticRegrModel.fit(X_train,y_train)
logisticPredictions = logisticRegrModel.predict(X_test)
PerformanceEvaluationMetrics(y_test,logisticPredictions)
#0.64 accuracy

#Building a Base SVM
'''svc= SVC(kernel='poly',random_state=2018)
svc.fit(X_train, y_train)
svm_y= svc.predict(X_test)
PerformanceEvaluationMetrics(y_test,svm_y)
0.61 accuracy
svc= SVC(kernel='sigmoid',random_state=2018)
svc.fit(X_train, y_train)
svm_y= svc.predict(X_test)
PerformanceEvaluationMetrics(y_test,svm_y)'''
#0.57 accuracy

svc= SVC(kernel='rbf',C=1,random_state=2018)
svc.fit(X_train, y_train)
svm_y= svc.predict(X_test)
PerformanceEvaluationMetrics(y_test,svm_y)
#0.65 Accuracy

# Instantiate RandomForrest
rf1 = RandomForestClassifier(random_state=2018,n_jobs=5,verbose=3,oob_score=True,n_estimators=999)
# Train the model on training data
rf1.fit(X_train,y_train);
#Predict y
y_pred=rf1.predict(X_test)
PerformanceEvaluationMetrics(y_test,y_pred)
#Accuracy 0.65

""" The Accuracy are more or less very similar for all the 3 models.
-SVM with rbf performs well means the data is non-linean
-SVM with rbf cannot identify the most predictive variables , so we eliminate svm """

'''-Feature Selection using Recursive elimination'''
Log_c_name,log_accuracy = RecursiveElimination(logisticRegrModel,X_train,y_train,X_test,y_test)
#after reduction in more than 31 variables the accuracy decrase gradually.We pick 31 variables for our model
rf1.model=RandomForestClassifier(random_state=2018,n_jobs=5,oob_score=True,n_estimators=99)
rf_c_name,rf_accuracy = RecursiveElimination(rf1.model,X_train,y_train,X_test,y_test)
#after reduction in more than 42 variables the accuracy decrase gradually.We pick 42 variables for our model

#Final Logistic Regression Model with 31 variable and Cross Validation to identify the optimal C value
Log_model_1=LogisticRegression(random_state = 2018)
Log_model_1=LogisticRegression(random_state = 2018)
rfe = RFE(logisticRegrModel,31)
rfe.fit(X_train,y_train)
X_train_log=X_train[:,rfe.support_]
X_test_log=X_test[:,rfe.support_]
params = {"C": [0.0005,0.01,0.03,0.05,0.5, 1,5,10,50,100]}
Log_model_cv = GridSearchCV(estimator = Log_model_1, param_grid = params, 
                        scoring= 'accuracy', 
                        cv = 10, 
                        verbose = 4,
                       return_train_score=True)
Log_model_cv.fit(X_train_log,y_train)
Log_model_cv.best_params_
Logistic_final_model = LogisticRegression(C = 1,random_state = 2018)
Logistic_final_model.fit(X_train_log,y_train)
logistic_final_Predictions = Logistic_final_model.predict(X_test_log)
PerformanceEvaluationMetrics(y_test,logistic_final_Predictions)
#Accuracy 0.64

#Checking for the P values
logm1 = sm.GLM(y_train,(sm.add_constant(X_train_log)), family = sm.families.Binomial(),C=1)
logm1.fit().summary()
#All the variables are having P value less than 5%

""" Feature importance for final logistic regression """

feat_importances_log=pd.DataFrame({'features':colnames[rfe.support_],'feature_coefficients': np.abs(Logistic_final_model.coef_).reshape(31,)})
feat_importances_log=feat_importances_log.sort_values('feature_coefficients')
feat_importances_log.plot.barh(x='features',y='feature_coefficients',figsize=(15,10))


""" Final Random Forest Model with 42 variable """

rfe1 = RFE(rf1.model,42)
rfe1.fit(X_train,y_train)
X_train_rf=X_train[:,rfe1.support_]
X_test_rf=X_test[:,rfe1.support_]
rf_model_cv=RandomForestClassifier(random_state=2018,n_jobs=5,oob_score=True,verbose=3)
param_grid = { 
    'n_estimators': [499,999,1499,1999],
    'max_features': ['sqrt','log2'],
    'criterion':['entropy','gini'],
    'min_samples_split' : [2,3,6,8]
}
CV_rfc = GridSearchCV(estimator=rf_model_cv, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train_rf,y_train)
CV_rfc.best_params_
rf_final_model = RandomForestClassifier(random_state=2018,n_jobs=5,oob_score=True,verbose=3,
                                        criterion='entropy',max_features='sqrt',min_samples_split=2,n_estimators=999)
rf_final_model.fit(X_train_rf,y_train)
rf_final_Predictions = rf_final_model.predict(X_test_rf)
PerformanceEvaluationMetrics(y_test,rf_final_Predictions)
#0.66

""" Feature importance for final random forest """

feat_importances_rf=pd.DataFrame({'features':colnames[rfe1.support_],'feature_importance':rf_final_model.feature_importances_})
feat_importances_rf=feat_importances_rf.sort_values('feature_importance')
feat_importances_rf.plot.barh(x='features',y='feature_importance',figsize=(15,10))

""" Since RandomForest has the highest accuracy we will pick the top predictors of gender from the same. """
