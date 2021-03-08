import glob
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize 
import pandas as pd
import numpy as np

root_dir = "../轉換後資料/POJ/"

files = []

for filename in glob.iglob(root_dir + '**/*.txt', recursive=True):
     files.append(filename)


texts = []

for path in files:
    data = open(path, encoding = 'utf-8').read()
    texts.append(data.lower())

corpus = ' '.join(texts)

tokenizer = RegexpTokenizer('[a-zA-Z0-9\\-]+|\\.|,|;|:|!|\\?')
tokenized_words = tokenizer.tokenize(corpus)

tokenized_sent = []
for i in sent_tokenize(corpus): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in tokenizer.tokenize(i): 
        temp.append(j.lower()) 
  
    tokenized_sent.append(temp) 




#print(tokenized_words[:10])
#print(tokenized_sent[:10])

fdist1 = nltk.FreqDist(tokenized_words)

#print(fdist1.most_common(20))

#print(fdist1['cha-pou'])



