from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Total.csv')
documents = df['track_en'].str.lower()

query = "treatment quarantine isolation fever city confirmed hospital patients wuhan diagnosed"

tvec = TfidfVectorizer(vocabulary=query.split(' '))
tvec_tfidf = tvec.fit_transform(documents)

tvec_tfidf_as_array = np.array(tvec_tfidf.toarray())
# print(tvec_tfidf_as_array)

good_documents = []
for d in tvec_tfidf_as_array:
    if sum(d) >= 0.5:
        #print((np.mean(d) * 2 ))
        good_documents.append(d)
## seanbon
# uniform_data = np.random.rand(10, 12)
# print(uniform_data)
ax = sns.heatmap(good_documents, linewidth=0)
#plt.subplots_adjust(left=0.2, right= 0.95)
plt.show()