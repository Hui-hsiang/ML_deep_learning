#%%
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing  import StandardScaler
from sklearn import preprocessing
import nltk
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

#%%
text = pd.read_csv("train.csv")

byauthor = text.groupby("author")

wordFreqByAuthor = nltk.probability.ConditionalFreqDist()

for name, group in byauthor:
   sentences = group['text'].str.cat(sep = ' ')
   sentences = sentences.lower()
   tokens = nltk.tokenize.word_tokenize(sentences)
   frequency = nltk.FreqDist(tokens)
   wordFreqByAuthor[name] = (frequency)

df_fdist = pd.DataFrame.from_dict(wordFreqByAuthor,orient='index')

col_name = df_fdist.columns

#%%
alist = []
for i in col_name:
   dropable = True
   for j in wordFreqByAuthor.keys():
      wordFreq = wordFreqByAuthor[j].freq(i)
      if wordFreq > 1e-3:
         dropable = False

   if dropable == False:
      alist.append(i)


#%%
xdf = pd.DataFrame(columns = alist)
ydf = pd.DataFrame(columns = ['author'] )
#%%
xdf = pd.read_csv("train_tree.csv")
ydf = pd.read_csv("train_author.csv")

#%%
for i in text.index :
   sentences = text.loc[i,'text']
   sentences = sentences.lower()
   tokens = nltk.tokenize.word_tokenize(sentences)
   xdf.loc[i] = 0
   print(i)
   for j in tokens:
      if j in alist:
         xdf.loc[i][j]+=1
   stra = text.loc[i,'author']
   ydf.loc[i] = stra

#%%   
xdf.to_csv("train_tree.csv")   
ydf.to_csv("train_author.csv")

xdf.info()
ydf.info()

#%%
X_train , X_test , y_train , y_test = train_test_split(xdf,ydf['author'],test_size=0.3,random_state=1010)


sc=StandardScaler()

sc.fit(X_train)
x_train_nor=sc.transform(X_train)
x_test_nor=sc.transform(X_test)

print (x_train_nor.shape)
print (x_test_nor.shape)
print (y_train.shape)
print (y_test.shape)

#%%
from keras.utils import to_categorical

for i in range(len(y_train)):
   if y_train.iloc[i] == 'EAP':
      y_train.iloc[i] = 0
   elif y_train.iloc[i] == 'HPL':
      y_train.iloc[i] = 1
   elif y_train.iloc[i] == 'MWS':
      y_train.iloc[i] = 2
for i in range(len(y_test)):
   if y_test.iloc[i] == 'EAP':
      y_test.iloc[i] = 0
   elif y_test.iloc[i] == 'HPL':
      y_test.iloc[i] = 1
   elif y_test.iloc[i] == 'MWS':
      y_test.iloc[i] = 2


y_train = pd.DataFrame(y_train) 
y_train.astype('int')
y_test = pd.DataFrame(y_test)
y_test.astype('int')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#%%
print (y_train)

#%%
def deep_model(i):
    model = Sequential()
    model.add(Dense(64, input_dim=141, activation='sigmoid'))
    for k in range(i):
      model.add(Dense(64,activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

for lavel in range(1,10):
   model = deep_model(lavel)
   model.fit(x_train_nor, y_train,epochs=25,verbose=1)
   y_test_predicted = model.predict(x_test_nor)
   print(y_test_predicted.shape[0])
   #  pdct = []
   #  for i in range(int(y_test_predicted.shape[0])):
   #      if (y_test_predicted[i,0] > y_test_predicted[i,1]):
   #          pdct.append(int(0))
   #      else:
   #          pdct.append(int(1))
   y_test_predicted = (y_test_predicted > 0.5)

   accuracy = accuracy_score(y_test, y_test_predicted)

   print('lavel = ',lavel,'training 準確率:',accuracy)
    
#%%
predata = pd.read_csv("test.csv")
for i in predata.index :
   sentences = predata.loc[i,'text']
   sentences = sentences.lower()
   tokens = nltk.tokenize.word_tokenize(sentences)
   testdf.loc[i] = 0
   for j in tokens:
      if j in testdf.columns:
         testdf.loc[i][j]+=1
testdf.to_csv("test_dt.csv")

output = clf.predict_proba(testdf)


submission = pd.read_csv("sample_submission.csv")
submission["EAP"] = output[:,0]
submission["HPL"] = output[:,1]
submission["MWS"] = output[:,2]
# submission[["EAP","HPL","MWS"]] = output[[:,0],[:,1],[0,2]]
submission.to_csv("submission.csv")

