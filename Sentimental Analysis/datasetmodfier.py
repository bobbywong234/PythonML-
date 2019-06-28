import pandas as pd
import numpy as np

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
  print(row['Translated_Review'],row['Sentiment'],i)
   
