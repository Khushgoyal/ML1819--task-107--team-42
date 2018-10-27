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


#Importing the Dataset for Model Building
twitter_data=pd.read_csv(r'C:\Users\johnp\Desktop\ML assignment\twitter_data_model_duilding.csv',sep=",",index_col='Unnamed: 0')

#Performace evaluation function for 
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

#Using Recursive feature elimination to try reducing the number of variables
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

#Twitter Data Scaling and Diving data into Test and Train
x = twitter_data.drop('gender',axis=1)
y = twitter_data.gender
colnames=x.columns 
sc = StandardScaler()  
X  = sc.fit_transform(x)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


###############################Logistic Regression###################################### 
#Logistic regression
# Make the instance of the model
#log_coeff=pd.DataFrame(model_col,columns=['log_coeff'])
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

svc= SVC(kernel='sigmoid',random_state=2018)
svc.fit(X_train, y_train)
svm_y= svc.predict(X_test)
PerformanceEvaluationMetrics(y_test,svm_y)'''

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