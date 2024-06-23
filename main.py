import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sbn


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sbn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()



if __name__ == "__main__":
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    
    X_train = train.loc[:,["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
    X_test = test.loc[:,["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
    Y_train = train.loc[:,["PassengerId","Survived"]]
    Y_test = test.loc[:,["PassengerId","Survived"]]
    
    X_train['Sex'] = X_train['Sex'].map({'female': 0, 'male': 1})
    X_test['Sex'] = X_test['Sex'].map({'female':0, 'male': 1})    
    
    # 新增特征Family Size
    X_train['Family Size'] = X_train['SibSp'] + X_train['Parch'] + 1
    X_test['Family Size'] = X_test['SibSp'] + X_test['Parch'] + 1
    
    # 对年龄,票价,SibSp,Parch进行标准化
    scaler = StandardScaler()
    X_train[['Age', 'Fare','SibSp','Parch','Family Size']] = scaler.fit_transform(X_train[['Age', 'Fare','SibSp','Parch','Family Size']])
    X_test[['Age', 'Fare','SibSp','Parch','Family Size']] = scaler.transform(X_test[['Age', 'Fare','SibSp','Parch','Family Size']])
    
    # 将train中age缺失的行删除，得到train_cleaned
    X_train_rowsWithNaN = X_train[X_train.isna().any(axis=1)]
    deleted_rows_index_train = X_train_rowsWithNaN.index
    X_train_cleaned = X_train.dropna()
    Y_train_cleaned = Y_train.drop(deleted_rows_index_train)
    train_cleaned = pd.concat([Y_train_cleaned,X_train_cleaned],axis=1,ignore_index=False)
    train_cleaned_dir = './train_cleaned.csv'
    train_cleaned.to_csv(train_cleaned_dir,index=False)
    
    # 将test中age缺失的行删除，得到test_cleaned
    X_test_rowsWithNaN = X_test[X_test.isna().any(axis=1)]
    deleted_rows_index_test = X_test_rowsWithNaN.index
    X_test_cleaned = X_test.dropna()
    Y_test_cleaned = Y_test.drop(deleted_rows_index_test)
    test_cleaned = pd.concat([Y_test_cleaned,X_test_cleaned],axis=1,ignore_index=False)
    test_cleaned_dir = './test_cleaned.csv'
    test_cleaned.to_csv(test_cleaned_dir,index=False)
    
    
    
    # 对test进行one-hot编码
    test_encoded = pd.get_dummies(test_cleaned,columns=['Pclass','Embarked'])
    test_encoded[['Pclass_1','Pclass_2','Pclass_3','Embarked_C','Embarked_Q','Embarked_S']] = test_encoded[['Pclass_1','Pclass_2','Pclass_3','Embarked_C','Embarked_Q','Embarked_S']].astype(int)
    test_encoded_dir = './test_encoded.csv'
    test_encoded.to_csv(test_encoded_dir,index=False)
    
    # 对train进行one-hot编码
    train_encoded = pd.get_dummies(train_cleaned,columns=['Pclass','Embarked'])
    train_encoded[['Pclass_1','Pclass_2','Pclass_3','Embarked_C','Embarked_Q','Embarked_S']] = train_encoded[['Pclass_1','Pclass_2','Pclass_3','Embarked_C','Embarked_Q','Embarked_S']].astype(int)
    train_encoded_dir = './train_encoded.csv'
    train_encoded.to_csv(train_encoded_dir,index=False)
    
    X_train_final = train_encoded.drop(['PassengerId','Survived','Fare'],axis=1)
    Y_train_final = train_encoded[['Survived']]
    X_test_final = test_encoded.drop(['PassengerId','Survived','Fare'],axis=1)
    Y_test_final = test_encoded[['Survived']]
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=10000, activation='relu', solver='adam', random_state=42)
    }
    
    plt.figure(figsize=(10, 7))
    
    for name, clf in classifiers.items():
        # 训练模型
        clf.fit(X_train_final, Y_train_final)
        
        # 进行预测
        Y_pred_prob = clf.predict_proba(X_test_final)[:, 1]
        Y_pred = clf.predict(X_test_final)
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(Y_test_final, Y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')
        
    
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    