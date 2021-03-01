from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('Total.csv')
documents = df['track_en'].str.lower()

query = "treatment quarantine isolation fever city confirmed hospital patients wuhan diagnosed"

# Vectorize every word into matrix on a table, transform to array for print
cvec = CountVectorizer(vocabulary=query.split(' '))
document_term_matrix = cvec.fit_transform(documents)
document_term_matrix_as_array = np.array(document_term_matrix.toarray())
print(document_term_matrix_as_array)



good_documents = []
for d in document_term_matrix_as_array:
    if sum(d) >= 10:
        good_documents.append(d)
## seanbon
#uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(good_documents, linewidth=0)
plt.show()
