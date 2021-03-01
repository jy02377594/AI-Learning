import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Automobile_data.csv', encoding='utf8')
x = dataset.iloc[1].values
y = dataset.iloc[2].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)