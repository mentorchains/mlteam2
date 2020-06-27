from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

my_csv = pd.read_csv('data.csv')# need data file at mlteam2/webcrawling/data.csv
corpus1 = list(my_csv.starter_content)#starter corpus including 3000 documents
corpus2 = list(my_csv.reply_content)#reply corpus including 3000 documents
corpust=corpus1[0:3]#run only for 3 post to make result easier to understand

vectorizer = CountVectorizer()
transformer = TfidfTransformer()  

wordcount1 = vectorizer.fit_transform(corpus1)#tokenization and calculate the word count matrix
tfidf1 = transformer.fit_transform(wordcount1)#calculate the tfidf matrix from word count matrix
word1 = vectorizer.get_feature_names() #get the word list index

# vectorizer = CountVectorizer()
# transformer = TfidfTransformer()

# wordcount2 = vectorizer.fit_transform(corpus2)
# tfidf2 = transformer.fit_transform(wordcount2)
# word2 = vectorizer.get_feature_names()

vectorizer = CountVectorizer()
transformer = TfidfTransformer()

wordcountt = vectorizer.fit_transform(corpust)#tokenization and calculate the word count matrix
tfidft = transformer.fit_transform(wordcountt) #calculate the tfidf matrix from word count matrix
wordt = vectorizer.get_feature_names()  # get the word list index

#for starter corpus
print(word1)
print("--------")
print(wordcount1)
print("--------")
print(tfidf1.toarray())

# only first 3 posts from starter corpus
print(wordt)
print("--------")
print(wordcountt)
print("--------")
print(tfidft.toarray())
