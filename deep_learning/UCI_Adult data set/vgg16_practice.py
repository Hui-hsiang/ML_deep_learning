 #%%
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing  import StandardScaler
from sklearn import preprocessing
import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import itertools
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
#%%
col_names = ['age','workclass','fnlwht','education','education-num','marital-status'
				   ,'occupation','relationship','race','sex','capital-gain','capital-loss',
				   'hours-per-week','native-country','result' ]
data = pd.read_csv("adult.csv",names = col_names)

data_clean = data.replace(regex=[r'\?|\.|\$'],value=np.nan)

adult = data_clean.dropna(how='any')

label_encoder = preprocessing.LabelEncoder()
for col in col_names:
    if (col in ['fnlwht','education-num','capital-gain','capital-loss','hours-per-week','age'] ):
        continue
    encoded = label_encoder.fit_transform(adult[col])
    adult[col] = encoded

#%%
X_train , X_test , y_train , y_test = train_test_split(adult[col_names[:14]],adult[col_names[14]],test_size=0.3,random_state=1010)
sc=StandardScaler()

sc.fit(X_train)
x_train_nor=sc.transform(X_train)
x_test_nor=sc.transform(X_test)
print (x_train_nor.shape)
print (y_train.shape)
y_train = to_categorical(y_train)
y_train.shape

#%% 
test_data = pd.read_csv("test.csv",names = col_names)
data_clean = test_data.replace(regex=[r'\?|\$'],value=np.nan)
test = data_clean.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
label_encoder = preprocessing.LabelEncoder()
for col in col_names:
    if (col in ['fnlwht','education-num','capital-gain','capital-loss','hours-per-week','age'] ):
        continue
    encoded = label_encoder.fit_transform(test[col])
    test[col] = encoded
xtest = test[col_names[:14]]
ytest = test[col_names[14]]

#%%
def deep_model(i):
    model = Sequential()
    model.add(Dense(64, input_dim=14, activation='sigmoid'))
    for k in range(i):
      model.add(Dense(64,activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

for lavel in range(1,10):
    model = deep_model(lavel)
    model.fit(x_train_nor, y_train,epochs=25,verbose=1)
    y_test_predicted = model.predict(x_test_nor)
    print(y_test_predicted.shape[0])
    pdct = []
    for i in range(int(y_test_predicted.shape[0])):
        if (y_test_predicted[i,0] > y_test_predicted[i,1]):
            pdct.append(int(0))
        else:
            pdct.append(int(1))
    accuracy = accuracy_score(y_test, pdct)

    print('lavel = ',lavel,'training 準確率:',accuracy)
    pdct = []  
    p = model.predict(xtest)


    for i in range(int(p.shape[0])):
        if (p[i,0] > p[i,1]):
            pdct.append(int(0))
        else:
            pdct.append(int(1))

    accuracy = accuracy_score(ytest, pdct)
    print('lavel = ',lavel,'testing 準確率:',accuracy)









# %%
