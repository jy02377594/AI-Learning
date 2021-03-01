import pandas as pd
from pandas import Series
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.svm import SVC# Support Vector Machine
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split


df_train = pd.read_csv("Wuhan_total_train_3.csv")
df_test = pd.read_csv("Wuhan_total_test_3.csv")

#################split directly
#df_total = pd.read_csv("Total_Wuhan&nonWuhan.csv", encoding='utf8')


##lower the word
#df_total["track"] = df_total["track"].str.lower()





#df_train, df_test = train_test_split(df_total, test_size=0.3)
#Seperate data into feature and results
X_train, y_train = df_train['track'].tolist(), df_train['label'].tolist()
X_test, y_test = df_test['track'].tolist(), df_test['label'].tolist()


#train model

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='linear', C=10, gamma= 0.1)),
])




# X_train = X_train_counts
# X_test = X_test_counts

text_clf.fit(X_train, y_train)

#predict class form test data
predicted = text_clf.predict(X_test)

#print(predicted)
print("Accuracy: {}%".format(text_clf.score(X_test, y_test) * 100 ))

print(metrics.classification_report(y_test, predicted))

print(y_test)

print(metrics.confusion_matrix(y_test, predicted))

df_test["predicted_class"] = Series(predicted, index=df_test.index)
df_test.to_csv("svm_predicted.csv", index=False)
