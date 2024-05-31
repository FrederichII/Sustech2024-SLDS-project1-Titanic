import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
test=pd.read_csv('./test.csv')
train=pd.read_csv('./train.csv')

X_train = train[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
X_test = test[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
Y_train = train["Survived"]
Y_test = test["Survived"]

X_train['Sex'] = X_train['Sex'].map({'female': 0, 'male': 1})
X_test['Sex'] = X_test['Sex'].map({'female':0, 'male': 1})

imputer = SimpleImputer(strategy='most_frequent')
X_train['Age'] = imputer.fit_transform(X_train[['Age']])
X_test[['Age']] = imputer.transform(X_test[['Age']])

X_train['Embarked'] = X_train['Embarked'].fillna('S')
X_test['Embarked'] = X_test['Embarked'].fillna('S')
# X_train['Embarked'] = X_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# X_test['Embarked'] = X_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X_train['Fare'] = imputer.fit_transform(X_train[['Fare']])
X_test['Fare'] = imputer.transform(X_test[['Fare']])

X_train.to_csv('X_train.csv',index=False)
Y_train.to_csv('Y_train.csv',index=False)
X_test.to_csv('X_test.csv',index=False)
Y_test.to_csv('Y_test.csv',index=False)

onehot_encoder = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', onehot_encoder, ['Embarked']),
    ],
    remainder='passthrough'  
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.fit_transform(X_test)

pd.DataFrame(X_train).to_csv('X_train_encoded.csv')
pd.DataFrame(X_test).to_csv('X_test_encoded.csv')

print(X_train)
print(X_test)





clf = DecisionTreeClassifier(
    criterion="entropy",
    splitter="best",
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features=None,
    random_state=42
)



pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', clf)
])



# 拟合模型
pipeline.fit(X_train, Y_train)
# print(model.coef_)
# print(model.intercept_)


# 将模型用于预测
Y_pred = pipeline.predict(X_test)
Y_pred = pd.DataFrame(Y_pred)
accuracy = accuracy_score(Y_test,Y_pred)
print(f'模型准确率: {accuracy * 100:.2f}%')

Y_pred_df = pd.DataFrame(Y_pred)
Y_pred_df.to_csv('Y_predict.csv',index=False)


# 网格搜索最优参数
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 3,4,5,6,7,8],
    'min_samples_split': [2,3,4,5,6,7],
    'min_samples_leaf': [1, 3, 5, 6, 7, 8],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, Y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))


# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(pipeline, X_train, Y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 30))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure()
plt.title("Learning Curve (Decision Tree Classifier)")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

# 绘制训练曲线
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

# 绘制交叉验证曲线
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()
