import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values
feature=X[:,4]   #taking only the feature from column four

hist_values,bin_edges=np.histogram(feature,bins=7)  #to get histogram for 7 buckets. chosen randomly.
plt.hist(feature,bins=7)
plt.xlabel("Feature value")   #x axis label
plt.ylabel("Frequency")    #y axis label
plt.show()

mean1=np.mean(feature)
variance1=np.var(feature)
print("Mean of feature is:",mean1)
print("Variance of feature is:",var1)
print("Frequency values:",hist_values)
print("Bin edges:",bin_edges)

