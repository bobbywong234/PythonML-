from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

df = pd.read_csv('APPSRev.csv')
df=df.drop(['App'],axis=1)
df=df.drop(['Sentiment_Polarity'],axis=1)
df=df.drop(['Sentiment_Subjectivity'],axis=1)

df=df.replace(np.nan,'',regex=True)

n=-1
i=0
for index,row in df.iterrows():
 n+=1
 if row['Translated_Review']=='' and row['Sentiment'] !='':
  i=n
  df=df.drop(labels=i)
  print(row['Translated_Review'],row['Sentiment'],'row',i,'Deleted')


reviews = df.iloc[:,0]
sentiment = df.iloc[:,1]

input=reviews.values.tolist()
labels=sentiment.values.tolist()
print('dataframe has converted into 2 lists')

input = list(filter(None,input))
labels = list(filter(None,labels))
print('All null values in input and labels have been removed')

for i in range(0,len(labels)):
 if labels[i]=='Positive':
  labels[i]=1
 if labels[i]=='Neutral':
  labels[i]=0
 if labels[i]=='Negative':
  labels[i]=-1

print('Sentiment Labels have changed, Positive ->1, Neutral ->0, Negative ->-1')

str(" ".join(input))
print('input flattened')

Tokenize = Tokenizer()
Tokenize.fit_on_texts(input)
Tokenized_input = Tokenize.texts_to_matrix(input,mode='tfidf')
print('Tokenized Matrix Constructed')

Hotcoded_input=[]
for sentence in input:
 words=text_to_word_sequence(sentence)
 vocabulary_size=len(words)
 if vocabulary_size == 1:
  vocabulary_size = 2
 
 Hotcoded_input.append(one_hot(sentence,vocabulary_size))


print('Sentence hot encoded')
lengths = []
for encoded_integer in Hotcoded_input:
 lengths.append(len(encoded_integer))

print('Max Length of Vector Getted')

padded_input = pad_sequences(Hotcoded_input,maxlen=max(lengths),padding='post')
print('The hot encoded data is really for importing')


import tensorflow as tf
from tensorflow.keras.layers import Conv1D,MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model

embedding_method = Embedding(max(lengths)+1,186,input_length=37427)
embedded_sequence = embedding_method(padded_input)
print('Embedding Layer Property:')
print(embedded_sequence)

Convlayer_1 = Conv1D(62,5,activation='relu')(embedded_sequence)
print('Convolutional Layer 1 Property:')
print(Convlayer_1)
print('')
Maxpooling_1 = MaxPooling1D(5)(Convlayer_1)
print('Maxpooling Layer 1 Property:')
print(Maxpooling_1)
print('')
Convlayer_2 = Conv1D(124,5,activation='relu')(Maxpooling_1)
print('Convolutional Layer 2 Property:')
print(Convlayer_2)
print('')
Maxpooling_2 = MaxPooling1D(5)(Convlayer_2)
print('Maxpooling Layer 2 Property:')
print(Maxpooling_2)
print('')
Convlayer_3 = Conv1D(186,5,activation='relu')(Maxpooling_2)
print('Convolutional Layer 3 Property:')
print(Convlayer_3)
print('')
Maxpooling_3 = MaxPooling1D(5)(Convlayer_3)
print('Maxpooling Layer 3 Property:')
print(Maxpooling_3)
print('')

Flatten_layer=Flatten()(Maxpooling_3)
print('Flatten Layer Property:')
print(Flatten_layer)
print('')
Dense_layer=Dense(186,activation='relu')(Flatten_layer)
print('FC Layer 1 Property:')
print(Dense_layer)
print('')
RNN_input_layer=Dense(2,activation='softmax')(Dense_layer)
print('FC Layer 2 Property:')
print(RNN_input_layer)

model=Model(padded_input,RNN_input_layer)
model.summary()
