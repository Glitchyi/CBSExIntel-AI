import re
from nltk.corpus import stopwords
from nltk.data import PathPointer
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

qsn=input("Enter quries separated by ';' ")

listified_qsn = qsn.split(';')

stop_words = set(stopwords.words('english'))
 

 
'''
filtered_list = [w for w in word_tokens if not w.lower() in stop_words]
 
filtered_list = []

# TODO: ask for queries and search to find Keywords
for w in word_tokens:
    if w not in stop_words:
        if w.isalpha():
            filtered_list.append(w)
filtered_list = [i.lower() for i in filtered_list]
filtered_list = [re.sub(" +",' ',data) for data in filtered_list]
filtered_str=""

for i in filtered_list:
    filtered_str+=f" {i}"

print(filtered_str,'\n')
'''
# set of documents
# instantiate the vectorizer object
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
# convert th documents into a matrix
tfidf_wm = tfidfvectorizer.fit_transform(listified_qsn)
#retrieve the terms found in the corpora
# if we take same parameters on both Classes(CountVectorizer and TfidfVectorizer) , it will give same output of get_feature_names() methods)
#count_tokens = tfidfvectorizer.get_feature_names() # no difference
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)

print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)