import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('APPSRev.csv')
df=df.drop(['App'],axis=1)
df=df.drop(['Sentiment_Polarity'],axis=1)
df=df.drop(['Sentiment_Subjectivity'],axis=1)

df=df.replace(np.nan,'',regex=True)

n=-1
i=0
dr=[]
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


encoded_input = [one_hot(word,50)for word in input]
print('input has been hot encoded to get unique vocabulary')
padded_input = pad_sequences(encoded_input,maxlen=4,padding='post')
print('The hot encoded 2d map has been sliced as minibatch and is really for inputting into the CNN training model')
  
  # import this script by using its file name
  # use exec(open('file_name.py').read()) to execute in python
  # look for index of the incomplete reviews in dataframe
  # and drop it 
  # so finnaly there's maxlen of 4 minibatch of hot encoded vocabulary made for traning CNN
   
