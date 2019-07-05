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


