#!/usr/bin/env python
# coding: utf-8

# # Parcial Corte 2
# 
# ## Hector Diaz 

# In[5]:


# Instalar NLP API's


# In[4]:


pip install -U spacy


# In[3]:


pip install -U click 


# In[ ]:


# Instalamos el modelo en Español para mayor comodidad 


# In[6]:


get_ipython().system('python3 -m spacy download es_core_news_sm')


# In[7]:


import nltk
nltk.download('stopwords')


# In[ ]:


# Importamos las librerias 


# In[8]:


import io
import sys
PATH = '/home/Elian,john y hector/Data'
DIR_DATA = '../Data/'
sys.path.append(PATH) if PATH not in list(sys.path) else None
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.express as px
# text API's
import re
import spacy
from sklearn.pipeline import FeatureUnion
import unicodedata
from nltk import TweetTokenizer
from spacy.lang.es import Spanish
from spacy.lang.en import English
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE,RandomOverSampler
from collections import Counter
from sklearn.preprocessing import LabelEncoder 
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


# In[9]:


# Inicializamos spacy y cargamos el modelo


# In[10]:


nlp = spacy.load('es_core_news_sm')


# In[ ]:


# Dataset


# In[11]:


data_raw = pd.read_csv(DIR_DATA + 'TASS2018.csv', sep=';')
data_raw.head(8)


# In[12]:


data_raw.info()


# In[13]:


# Descripción del tiempo de tweets y polaridad 


data_raw['hour'] = pd.DatetimeIndex(data_raw['date']).hour
data_raw['minute'] = pd.DatetimeIndex(data_raw['date']).minute
fig = px.line(data_raw, x='polarity', y='hour') 
fig.show()


tweet_by_polarity = data_raw.groupby("polarity", as_index=False)['content'].count()
tweet_by_polarity.head(10)


# In[14]:


plt.figure(figsize=(15,5))
sns.barplot(tweet_by_polarity['polarity'].values, tweet_by_polarity['content'].values, alpha=0.8)
plt.title('Polary Frequency by content')
plt.ylabel('content', fontsize=10)
plt.xlabel('polarity', fontsize=10)
plt.show()


# In[15]:


#Uso de palabras en tweets

words = {}
for row in tqdm(data_raw['content']):
    doc = nlp(row.lower())
    for token in doc:
        if token.is_alpha and not token.is_stop:
            if token.text in words:
                num_temp = int(words[token.text])
                words[token.text] = num_temp + 1
            else:
                words[token.text] = 1


# In[16]:


del words['a']
df_words = pd.DataFrame([[key, words[key]] for key in words.keys()], columns=['Word', 'Freq'])
df_words.sort_values('Freq').tail(15)
df_words = df_words[:20]
df_words.head(10)


# In[ ]:





# In[17]:


pos_freq = {}
for row in tqdm(data_raw['content'].to_list()):
    doc = nlp(row.lower())
    for token in doc:
        if token.pos_ in pos_freq:
            value = pos_freq[token.pos_]
            pos_freq[token.pos_] = value + 1
        else:
            pos_freq[token.pos_] =  1
            


# In[22]:


plt.figure(figsize=(15,5))
sns.barplot(df_words['Word'].values, df_words['Freq'].values, alpha=0.8)
plt.title('Word Frequency')
plt.ylabel('Freq', fontsize=10)
plt.xlabel('Word', fontsize=10)
plt.show()


# In[24]:


# Procesamiento 

def processing(text: str):
    result = ''
    try:
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text_out = text.decode("utf-8")
        text_out = text_out.lower()
        text_out = re.sub("[\U0001f000-\U000e007f]", 'EMOJI', text_out)
        text_out = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                          r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                          'URL', text_out)
        text_out = re.sub("@([A-Za-z0-9_]{1,40})", 'MENTION', text_out)
        text_out = re.sub("#([A-Za-z0-9_]{1,40})", 'HASTAG', text_out)
        # Remove patterns
        
        text_out = re.sub(r'\©|\×|\⇔|\_|\»|\«|\~|\#|\$|\€|\Â|\�|\¬', '', text_out)
        text_out = re.sub(r'\,|\;|\:|\!|\¡|\’|\‘|\”|\“|\"|\'|\`', '', text_out)
        text_out = re.sub(r'\}|\{|\[|\]|\(|\)|\<|\>|\?|\¿|\°|\|', '', text_out)
        text_out = re.sub(r'\/|\-|\+|\*|\=|\^|\%|\&|\$', '', text_out)
        text_out = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text_out)
        text_out = re.sub(r'[0-9]', '', text_out)
        text_out = re.sub(r'\s+', ' ', text_out).strip()
        text_out = text_out.rstrip()
        result = text_out if text_out != ' ' else None
    except Exception as e:
        print('Error processing: {0}'.format(e))
    return result


# In[25]:


messages = [processing(row) for row in data_raw['content'].tolist()]
messages


# # Union de caracteristicas 

# In[26]:


stop_words = set(stopwords.words('spanish'))


# In[27]:


bow = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words=stop_words)

tf=TfidfVectorizer()

union=FeatureUnion([
    ('bow_vector', bow),
    ('tfidf_vector', tf)
])


# In[28]:


union.fit(messages)


# In[29]:


x = union.transform(messages).toarray()


# In[30]:


df = pd.DataFrame(x, index=['tweet '+str(i) for i in range(1, 1+len(messages))], columns=union.get_feature_names())
df


# In[31]:


y = data_raw['polarity']
y


# In[32]:


Counter(y)


# In[34]:


#uso de randomoversampler en vez de smoth
oversample = RandomOverSampler()
x, y = oversample.fit_resample(x, y)


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[37]:


softmax = LogisticRegression(multi_class="multinomial", solver="lbfgs")

softmax.fit(x_train, y_train)


# In[38]:


y_predict = softmax.predict(x_test)
y_predict


# In[39]:


plot_confusion_matrix(softmax, x_test, y_test) 
plt.show()  


# In[40]:


f1 = f1_score(y_test, y_predict, average="macro")
precision = precision_score(y_test, y_predict, average="macro")
recall = recall_score(y_test, y_predict, average="macro")
accuracy = accuracy_score(y_test, y_predict, normalize=True)
print('F1: ',f1)
print('Precision: ', precision)
print('Recall: ', recall)
print('Accuracy: ', accuracy)


# In[ ]:




