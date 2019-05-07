import os
import numpy as np
import pandas as pd
import sys
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


nltk.download('wordnet')
nltk.download('stopwords')

np.sqrt(2)
print(sys.version)

dir_path = '../dcdir/data/NetworkData/'
person_dir = os.path.join(dir_path, "persons.csv")
persons_raw = pd.read_csv(person_dir, index_col=0)
product_dir = os.path.join(dir_path, "only_these_products.csv")
products_idx = pd.read_csv(product_dir, header=None).loc[:, 0]
assert len(products_idx.unique()) == len(products_idx)
person_product_dir = os.path.join(dir_path, "personProducts.csv")
product_dir = os.path.join(dir_path, "products.csv")
person_products = pd.read_csv(person_product_dir, index_col=False)
products_raw = pd.read_csv(product_dir, index_col=False)

usual_products_df = person_products[person_products.groupby("productId")["productId"].transform('size') > 9]
usual_product_ids = usual_products_df.productId
products = products_raw.loc[products_raw.id.isin(usual_product_ids)][['id', 'description', 'title']]
products.description.fillna(products.title, inplace=True)
products.title.fillna(products.description, inplace=True)
products['text'] = products.description + ' ' + products.title
products = products.drop(['description', 'title'], axis=1)

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
def rem_punct(text):
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    return text

def rem_num(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def drop_duplicates(text):
    words = text.split()
    return ' '.join(np.unique(words).tolist())


def get_lemma(word):
    return WordNetLemmatizer().lemmatize(word)


def preprocess(text):
    text = rem_punct(str(text))
    text = rem_num(text)
    text = text.lower()
    text = drop_duplicates(text)
    token_list = word_tokenize(text)
    token_list = [token for token in token_list if token not in stop_words]
    token_list = [token for token in token_list if len(token) > 2]
    token_list = list(set([get_lemma(token) for token in token_list]))
    token_list = list(set([porter.stem(token) for token in token_list]))
    return token_list


# data = products['text'].astype(str).apply(preprocess) q
# print(preprocess(data.iloc[0]))
data = []
text = products['text']
for i in range(0, 3, 1):
    temp = text.iloc[i]
    temp = preprocess(temp)
    data.append(temp)

print(data)
