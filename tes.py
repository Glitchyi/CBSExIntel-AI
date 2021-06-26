import re
from nltk.corpus import stopwords
from nltk.data import PathPointer
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


example_sent = """An apple is an edible fruit produced by an apple tree (Malus domestica). Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found today. Apples have been grown for thousands of years in Asia and Europe and were brought to North America by European colonists. Apples have religious and mythological significance in many cultures, including Norse, Greek, and European Christian tradition.

Apple trees are large if grown from seed. Generally, apple cultivars are propagated by grafting onto rootstocks, which control the size of the resulting tree. There are more than 7,500 known cultivars of apples, resulting in a wide range of desired characteristics. Different cultivars are bred for various tastes and use, including cooking, eating raw and cider production. Trees and fruit are prone to a number of fungal, bacterial and pest problems, which can be controlled by a number of organic and non-organic means. In 2010, the fruit's genome was sequenced as part of research on disease control and selective breeding in apple production.

Worldwide production of apples in 2018 was 86 million tonnes, with China accounting for nearly half of the total.[3]"""
 
stop_words = set(stopwords.words('english'))
 
word_tokens = word_tokenize(example_sent)
 
filtered_list = [w for w in word_tokens if not w.lower() in stop_words]
 
filtered_list = []

# TODO ask for queries and search to find Keywords

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
# set of documents
train = ['The sky is blue.','The sun is bright.']
test = ['The sun in the sky is bright', 'We can see the shining sun, the bright sun.']
# instantiate the vectorizer object
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
# convert th documents into a matrix
tfidf_wm = tfidfvectorizer.fit_transform(train)
#retrieve the terms found in the corpora
# if we take same parameters on both Classes(CountVectorizer and TfidfVectorizer) , it will give same output of get_feature_names() methods)
#count_tokens = tfidfvectorizer.get_feature_names() # no difference
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = ['Doc1','Doc2'],columns = tfidf_tokens)

print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)