""" * Gender prediction using Twitter data *








"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import cross_val_score


#Importing the Dataset for Model Building
twitter_data=pd.read_csv(r'C:\Users\johnp\Desktop\ML assignment\twitter_data_model_duilding.csv',sep=",",index_col='Unnamed: 0')


#Removing features that have correlation of 0.75 or more
new_twitter_data= twitter_data.drop('gender',axis=1)
# Create correlation matrix
corr_matrix = new_twitter_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.70)]


twitter_data=twitter_data.drop(to_drop,axis=1)
#Correlation Plot
f, ax = plt.subplots(figsize=(8, 8))
corr = new_twitter_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


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
