import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif,f_classif
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LeakyReLU,BatchNormalization
from keras.optimizers import SGD,Adam
from keras.utils import to_categorical

input_dim=139
output_dim=10
droupout_rate=0.4
neurons=300
epochs=300
batch_size=512
def bar(X_train,Y_train,X_unlabeled,Y_sude,bar,sample_weight,scale):
    num=0
    Xt=[]
    Yt=[]
    Xu=[]
    for id,ys in enumerate(np.max(Y_sude,1)):
        if ys>=bar:
            Xt.append(X_unlabeled[id,:])
            Yt.append(np.argmax(Y_sude[id,:]))
            num=num+1
        else:
            Xu.append(X_unlabeled[id,:])
    X_train=np.vstack((X_train,np.array(Xt)))
    Y_train=np.hstack((Y_train,np.array(Yt)))
    Xu=np.array(Xu)
    sample_weight=np.hstack((sample_weight,np.ones(num)*scale))
    return X_train,Y_train,num,Xu,sample_weight
def get_model(input_dim=120,output_dim=5,droupout_rate=0.5,neurons=300,epochs=300,batch_size=10000):
    model=Sequential()
    model.add(Dense(neurons,input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(droupout_rate))
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(droupout_rate))
    model.add(Dense(neurons//2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(droupout_rate))
    model.add(Dense(neurons//2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(droupout_rate))
    model.add(Dense(neurons//2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(droupout_rate))
    model.add(Dense(output_dim,activation='softmax'))
    return model

df_train_labeled = pd.read_hdf("train_labeled.h5","train")
df_train_unlabeled = pd.read_hdf("train_unlabeled.h5","train")
df_test = pd.read_hdf("test.h5","test")
X_train = np.array(df_train_labeled.iloc[:,1:140])

Y_train = np.array(df_train_labeled.iloc[:,0])
X_train,Y_train = shuffle(X_train,Y_train)
#X_val=X_train[8000:,:]
#Y_val=Y_train[8000:]
#X_train=X_train[0:8000,:]
#Y_train=Y_train[0:8000]
print(X_train.shape)
print(Y_train.shape)
#print(X_val.shape)
#print(Y_val.shape)
#X_val0=X_val
#Y_val0=Y_val

X_unlabeled = np.array(df_train_unlabeled.iloc[:,0:139])
print(X_unlabeled.shape)
X_test = np.array(df_test.iloc[:,0:139])
sample_weight=np.ones(Y_train.shape)
scale=1
num=10000000000

model=get_model(input_dim=input_dim,output_dim=10,neurons=600,droupout_rate=0.4)

lr=0.01
while(num>=500):
    #sgd = SGD(lr=lr, decay=1e-6, momentum=0.995, nesterov=True)
    adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
              optimizer=adm,
              metrics=['accuracy'])
    X_unlabeled0=X_unlabeled
    X_train0 = X_train
    Y_train0=Y_train
    #X_val=X_val0
    #Y_val=Y_val0

    ss = StandardScaler()
    #ss=MinMaxScaler()
    X_train = ss.fit_transform(X_train)
    X_unlabeled=ss.transform(X_unlabeled)
    #X_val=ss.transform(X_val)
    #X_test=ss.transform(X_test)

    #selector=SelectKBest(f_classif, k=input_dim)
    #X_train = selector.fit_transform(X_train, Y_train)
    #ss = StandardScaler()
    #ss=MinMaxScaler()
    #
    Y_train=to_categorical(Y_train)
    #Y_val=to_categorical(Y_val)
    model.fit(X_train, Y_train,epochs=epochs,batch_size=batch_size, class_weight = 'auto',sample_weight=sample_weight)
    Y_sude=model.predict(X_unlabeled)
    scale=scale*0.8
    X_train,Y_train,num,X_unlabeled,sample_weight=bar(X_train0,Y_train0,X_unlabeled0,Y_sude,0.99,sample_weight,scale)
    print(num)
    epochs-=50
    lr-=0.0025
model.fit(X_train, Y_train,epochs=500,batch_size=batch_size, class_weight = 'auto',sample_weight=sample_weight)
X_test=ss.transform(X_test)
y_predit = model.predict(X_test)
y_predit = np.argmax(y_predit,axis=1)
row=y_predit.shape[0]
seq=np.array(range(30000,30000+row))
data = pd.DataFrame({"Id":seq,"y":y_predit})
data.to_csv('./res.csv',index=False,header=True)