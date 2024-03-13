import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC


x_train = pd.read_csv(r"./data/x_train.csv")
y_train = pd.read_csv(r"./data/y_train.csv")
x_test = pd.read_csv(r"./data/x_test.csv")

data = pd.merge(x_train, y_train, on = "ID", how = "inner")




counter = Counter(data['intention'].tolist())
top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
data = data[data['intention'].map(lambda x: x in top_10_varieties)]

description_list = data['question'].tolist()
varietal_list = [top_10_varieties[i] for i in data['intention'].tolist()]
varietal_list = np.array(varietal_list)

count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(description_list)


tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, varietal_list, test_size=0.3)

clf = SVC(kernel='linear').fit(train_x, train_y)
y_score = clf.predict(test_x)

n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))


x_test_counts = count_vect.transform(x_test['question'].tolist())
x_test_tfidf = tfidf_transformer.transform(x_test_counts)


y_test_score = clf.predict(x_test_tfidf)

x_test['predicted_intention'] = [list(top_10_varieties.keys())[i] for i in y_test_score]

output_test = x_test[["ID", "predicted_intention"]]


output_test.to_csv("submission.csv", index=False)

print("Le projet est bien enregistrer en submission.csv")
