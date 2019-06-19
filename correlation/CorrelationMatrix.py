import pandas as pd
import numpy as np
import seaborn as seb
import matplotlib.pyplot as plot

df=pd.read_csv('MLdataset.csv')
df=df.drop(['User'],axis=1)
df=df.drop(['Unnamed: 25'],axis=1)
correlation = df.astype(np.float64).corr()
mask=np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
camp = seb.diverging_palette(240,10,n=7,as_cmap=True)
cm=seb.heatmap(correlation, mask=mask, cmap=camp, vmax=1, vmin=-1,center=0,square=True,linewidths=.3)
plot.title('Correlation Matrix - The Coefficiency of each pair of Locations')
plot.show()

