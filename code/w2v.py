import glob
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize 
import pandas as pd
import numpy as np
import re

root_dir = "../轉換後資料/POJ/"

def pre_process(root_dir):

    files = []

    for filename in glob.iglob(root_dir + '**/*.txt', recursive=True):
         files.append(filename)


    texts = []

    for path in files:
        data = open(path, encoding = 'utf-8').read()
        texts.append(data)

    corpus = ' '.join(texts)

    tokenizer = RegexpTokenizer('[a-zA-Z0-9\\-]+|\\.|,|;|:|!|\\?')
    tokenized_words = tokenizer.tokenize(corpus)

    tokenized_sent = []
    for i in sent_tokenize(corpus): 
        temp = [] 
          
        # tokenize the sentence into words 
        for j in tokenizer.tokenize(i): 
            temp.append(re.sub("N([0-9]|$|-)", "nn\g<1>", j).lower()) 
      
        tokenized_sent.append(temp) 

    return tokenized_sent



#tokenized_sent = pre_process(root_dir)
#print(tokenized_sent[200:300])


####test word2vec
import gensim
from gensim.models import Word2Vec 




test = "this is a book."
#print(word_tokenize(test))
#print([x for x in tokenized if x == "i-seng"])


#model1 = gensim.models.Word2Vec(tokenized_sent, min_count = 10,  
#                             size = 300, window = 5) 


model1 = Word2Vec.load("word2vec.model")

def query(word):
    print(word)
    print("cha-pou")
    print(model1.wv.similarity(word, 'cha-pou'))
    print("cha-bou2")
    print(model1.wv.similarity(word, 'cha-bou2'))
    print("#####")



query("i-seng")

query("hou7-su7")

query("sian-sinn")

query("sian-senn")

query("lau7-su")

query("chong2-thong2")

query("thau5-ke")


query("pa-pa")




exit()








model1.save("word2vec.model")

vocab, vectors = model1.wv.vocab, model1.wv.vectors

# get node name and embedding vector index.
name_index = np.array([(v[0], v[1].index) for v in vocab.items()])

# init dataframe using embedding vectors and set index as node name
df =  pd.DataFrame(vectors[name_index[:,1].astype(int)])
df.index = name_index[:, 0]
df.to_csv("embedding.csv")