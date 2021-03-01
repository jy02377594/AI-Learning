from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import codecs
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Total.csv')
documents = df['track_en'].str.lower()

query = "treatment quarantine isolation fever city confirmed hospital patients wuhan diagnosed"
# Vectorize every word into matrix on a table, fit_transform include training and transform
cvec = CountVectorizer(vocabulary=query.split(' '))
document_term_matrix = cvec.fit_transform(documents)

##  Get the total documents number
track_count=0
f = codecs.open('Total.csv', 'rU', 'utf-8')
for line in f:
    track_count +=1
print("track_count",track_count)

# get the document frequency from collections import default dict
import math
##  Get the documents frequency
DF = [0,0,0,0,0,0,0,0,0,0]
IDF=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
array_count=0


array_query=["treatment","quarantine"," isolation"," fever "," city"," confirmed "," hospital "," patients "," wuhan "," diagnosed "]
while array_count<len(array_query):
    with codecs.open('Total.csv', 'rU', 'utf-8') as ins:
        for line in ins:
            if array_query[array_count] in line.lower():
                DF[array_count] += 1
            else:
                continue
            IDF[array_count] = math.log(float(track_count) / float(DF[array_count]))
    array_count +=1

print("DF",DF)
print("IDF",IDF)

## plot
fig, ax = plt.subplots()
width = 0.56 # the width of the bars
ind = np.arange(len(IDF))  # the x locations for the groups
ax.barh(ind, IDF, width, color="#9a96a3")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(array_query, minor=False)

plt.subplots_adjust(left=0.2, right= 0.95)
plt.xlabel("IDF")
plt.ylabel("query terms")
plt.title("IDF Bar Chart")
plt.show()





