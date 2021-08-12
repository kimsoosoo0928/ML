from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor
import pandas as pd
import numpy as np

# 1. data
datasets = load_iris()
df_iris = pd.DataFrame(datasets.data,columns=datasets.feature_names)
df_iris['target'] = pd.Series(datasets.target)
df_iris = df_iris.drop(['petal length (cm)', 'petal width (cm)'], axis=1)
datasets = np.array(df_iris)


x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

# 2. model
model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

# 3. fit
model.fit(x_train, y_train)

# 4. eval, pred
acc = model.score(x_test, y_test)
print('acc : ', acc)

'''
acc :  0.9333333333333333
'''

print(model.feature_importances_) # 강력함
# [0.         0.0125026  0.03213177 0.95536562]
# iris의 컬럼은 4개, 전부 더하면 1
# 첫번째 칼럼은 acc를 만드는데 도움을 주지 않았다.


# 그림 그리기

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

'''
DecisionTreeClassifier original
acc :  0.9333333333333333
[0.0125026  0.         0.53835801 0.44913938]

'''