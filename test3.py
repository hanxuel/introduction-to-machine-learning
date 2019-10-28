import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif,f_classif
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LeakyReLU,BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical

input_dim=120
output_dim=5
droupout_rate=0.4
neurons=300
epochs=500
batch_size=3000
def get_model(input_dim=120,output_dim=5,droupout_rate=0.5,neurons=300,epochs=300,batch_size=10000):
    model=Sequential()
    model.add(Dense(neurons,input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim,activation='softmax'))
    return model
data_train = pd.read_hdf("train.h5", key="train")
data_test = pd.read_hdf("test.h5", key="test")
X_train=data_train.iloc[:,1:]
X_train=np.array(X_train)
Y_train=data_train.iloc[:,0]
Y_train=np.array(Y_train)
#Y_train=to_categorical(Y_train)
X_test=np.array(data_test)
print(X_train.shape,X_test.shape)
ss = StandardScaler()
#ss=MinMaxScaler()
X_train = ss.fit_transform(X_train)
X_test=ss.transform(X_test)

selector=SelectKBest(f_classif, k=input_dim)
X_train = selector.fit_transform(X_train, Y_train)
#ss = StandardScaler()
#ss=MinMaxScaler()
#X_train = ss.fit_transform(X_train)

Y_train=to_categorical(Y_train)
model=get_model(input_dim=input_dim,output_dim=5,neurons=512)
sgd = SGD(lr=0.04, decay=1e-6, momentum=0.99, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, Y_train,epochs=epochs,batch_size=batch_size,validation_split=0.1, shuffle=True, class_weight = 'auto')
X_test=selector.transform(X_test)
model.fit(X_train, Y_train,epochs=epochs,batch_size=batch_size,class_weight = 'auto')
Y_test=model.predict(X_test)
print(Y_test)
print(Y_test.shape)
print(np.argmax(Y_test, axis=1))
Y_test=np.argmax(Y_test, axis=1)
data={'ID':np.arange(45324,45324+Y_test.shape[0],1),'y':Y_test}
data=pd.DataFrame(data)
data.to_csv('model512.csv',index=False)
