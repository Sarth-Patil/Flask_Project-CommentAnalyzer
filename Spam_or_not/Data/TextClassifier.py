import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

df1 = pd.read_csv("Youtube01-Psy.csv")
df2 = pd.read_csv("Youtube02-KatyPerry.csv")
df3 = pd.read_csv("Youtube03-LMFAO.csv")
df4 = pd.read_csv("Youtube04-Eminem.csv")
df5 = pd.read_csv("Youtube05-Shakira.csv")

dfs = [df1,df2,df3,df4,df5]
df_merged = pd.concat(dfs)
keys = ["Psy","KatyPerry","LMFAO","Eminem","Shakira"]
df_with_keys = pd.concat(dfs,keys=keys)


df_with_keys.to_csv("Merged_data.csv")
df = pd.read_csv("Merged_data.csv", encoding='latin-1')

df_data = df[["CONTENT","CLASS"]]
df_x = df_data['CONTENT']
df_y = df_data['CLASS']


corpus = df_x
cv = CountVectorizer()
X = cv.fit_transform(corpus)

X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

naivebayesML = open("model.pkl","wb")
pickle.dump(clf,naivebayesML)
naivebayesML.close()