# Import data
import pandas as pd
data=pd.read_csv("SMSSpamCollection", sep="\t", names=["Label","Messages"])

# Cleaning and preprocessing

import re
import nltk
from nltk.corpus import stopwords 

# Stemming

from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

corpus_stemm=[]
for i in range(0, len(data)):
    punc=re.sub("[^a-zA-Z]", " ", data["Messages"][i])
    lower_case=punc.lower()
    words=lower_case.split()
    words=[stemmer.stem(word) for word in words if not word in stopwords.words("english")]
    words=" ".join(words)
    corpus_stemm.append(words)
    
# Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x=cv.fit_transform(corpus_stemm).toarray()

y=pd.get_dummies(data["Label"])
y=y.iloc[:,1].values

# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size = 0.20, random_state = 0)

# Naive bayes Classifier
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(x_train,y_train)

y_pred_stemm=model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm_stemm=confusion_matrix(y_test, y_pred_stemm)

from sklearn.metrics import accuracy_score
accu_score_stemm=accuracy_score(y_test, y_pred_stemm)

# lemmitization

from nltk.stem import WordNetLemmatizer
lemmitizer=WordNetLemmatizer()

corpus_lemm=[]
for i in range(0, len(data)):
    punc=re.sub("[^a-zA-Z]", " ", data["Messages"][i])
    lower_case=punc.lower()
    words=lower_case.split()
    words=[lemmitizer.lemmatize(word) for word in words if not word in stopwords.words("english")]
    words=" ".join(words)
    corpus_lemm.append(words)
    
# Usine TF-IDF instead of bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf=TfidfVectorizer(max_features=5000)
x=tf_idf.fit_transform(corpus_lemm).toarray()

model_lemm=MultinomialNB().fit(x_train,y_train)
y_pred_lemm=model_lemm.predict(x_test)

from sklearn.metrics import confusion_matrix
cm_lemm=confusion_matrix(y_test, y_pred_lemm)

from sklearn.metrics import accuracy_score
accu_score_lemm=accuracy_score(y_test, y_pred_lemm)
 