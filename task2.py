import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from minepy import MINE
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)

data_train=pd.read_csv('train.csv')
X_train=data_train.iloc[:,2:22]
X_train=np.array(X_train)
Y_train=data_train.iloc[:,1]
Y_train=np.array(Y_train)
selector=SelectKBest(mutual_info_classif, k=12)
X_train = selector.fit_transform(X_train, Y_train)
ss = StandardScaler()
#ss=MinMaxScaler()
X_train = ss.fit_transform(X_train)

accuracy=0
for i in range(1,21):
    seed=np.random.randint(0,100)
    test_size = 0.2
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, Y_train, test_size=test_size, random_state=seed)
    model=SVC(C=17.0, kernel='rbf', degree=5, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
    #model = XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=100,objective='multi:softmax')
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    predictions = [round(value) for value in y_pred]
    accuracy =accuracy+ accuracy_score(y_te, predictions)

print("Accuracy: %.2f%%" % (accuracy * 5.0))
#model = XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=100,objective='multi:softmax')
model=SVC(C=17.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
model.fit(X_train, Y_train)

data_test=pd.read_csv('test.csv')
X_test=data_test.iloc[:,1:21]
X_test=np.array(X_test)
X_test=selector.transform(X_test)
X_test=ss.transform(X_test)
y_pred=model.predict(X_test)
y_pred=y_pred.astype(int)
result=np.zeros((y_pred.shape[0],2))
result[:,0]=range(2000,2000+y_pred.shape[0])
result[:,1]=y_pred
resdf = pd.DataFrame(result)
resdf.columns=['Id','y']
resdf.to_csv('result.csv',index=False)
