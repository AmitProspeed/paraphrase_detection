import pandas as pd
import numpy as np
import nltk
import time
import re
import unicodedata
import contractions
import string
import math
import random
import spacy
spacy.load('en_core_web_lg')
import en_core_web_lg
import gensim
import gzip

from scipy import spatial
from scipy.spatial.distance import cosine

from num2words import num2words

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from spacy import displacy
from collections import Counter
from tqdm import tqdm_notebook
from sense2vec import Sense2VecComponent

from google.colab import drive
drive.mount('/content/drive')

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

stopwords1 = stopwords.words('english')


def preprocessing_pipeline(q1):
    q1 =  unicodedata.normalize('NFKD', q1).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    q1 = re.sub(r'[^\x00-\x7f]', r'', q1)
    q1 = "".join([c.lower() for c in q1 if c not in string.punctuation])
    q1 = word_tokenize(q1)
    q1 = [num2words(i) if i.isdigit() else i for i in q1]
#   q1 = list(filter(lambda x:x not in stopwords, q1))
    q1 = " ".join(list(map(lambda x: contractions.fix(x), q1)))
    return(q1)

def tokenize1(text):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def basic_nlp_features(q1, q2, w1, w2, q1p, q2p):
    q1_len_with_space, q2_len_with_space  = len(q1), len(q2)
    q1_len_without_space, q2_len_without_space  = len(re.sub(' ','',q1)), len(re.sub(' ','',q2))
    if len(w1)==0 or len(w2)==0:
        return 0,0,0,0,0,0,0,0,0,0
    q1_no_of_uniq_words, q2_no_of_uniq_words = len(set(w1)), len(set(w2))
    diff_len_with_space = q1_len_with_space - q2_len_with_space
    diff_len_without_space = q1_len_without_space - q2_len_without_space
    diff_no_of_words = len(w1) - len(w2)
    diff_no_of_uniq_words = q1_no_of_uniq_words - q2_no_of_uniq_words
    return q1_len_with_space, q2_len_with_space, q1_len_without_space, q2_len_without_space, q1_no_of_uniq_words, q2_no_of_uniq_words, diff_len_with_space, diff_len_without_space, diff_no_of_words, diff_no_of_uniq_words

def bag_of_words(df, ngram_flag):
    # Experiment with min_df and max_df
    if ngram_flag:
        vectorizer = CountVectorizer(tokenizer=tokenize1, stop_words=stopwords1, ngram_range=(1, 2))
    else:
        vectorizer = CountVectorizer(tokenizer=tokenize1, stop_words=stopwords1)
    x = vectorizer.fit_transform(df)
    return  vectorizer


def tfidf(df):
    vectorizer = TfidfVectorizer(tokenizer=tokenize1, stop_words=stopwords1)
    x = vectorizer.fit_transform(df)
    return vectorizer

def trigram(df):
    vectorizer = CountVectorizer(tokenizer=tokenize1, stop_words=stopwords1, ngram_range=(3, 3))
    x = vectorizer.fit_transform(df)
    return (vectorizer)

def idf(df):
    vectorizer = TfidfVectorizer(tokenizer=tokenize1, stop_words=stopwords1, use_idf=True)
    x = vectorizer.fit_transform(df)
    return vectorizer

def named_entity_features(q1, q2):
    q1p = nlp(q1)
    q2p = nlp(q2)
    ent1 = [X.text for X in q1p.ents]
    ent2 = [X.text for X in q2p.ents]
    common = set(ent1).intersection(set(ent2))
    different = set(ent1).union(set(ent2)) - common
    matching_ne = len(common)
    non_matching_ne = len(different)
    return(matching_ne, non_matching_ne)

def word_match_share(w1, w2):
    q1words = {}
    q2words = {}
    for word in w1:
        q1words[word] = 1
    for word in w2:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    match_score = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))
    return match_score

def first_word_similarity(w1, w2):
    fw_q1 = w1[0]
    fw_q2 = w2[0]
    if(len(w1)==0 or len(w2)==0):
      return 0
    if fw_q1 == fw_q2 and fw_q1 in question_types:
      return 1
    else:
      return 0

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1.0 / (count + eps)

def tfidf_word_match(w1, w2, q1p, q2p):
    q1words = {}
    q2words = {}
    for word in w1:
        q1words[word] = 1
    for word in w2:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0 :
        return 0,0,0,0,0,0,0,0,0,0,0
    words = q1p + q2p
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    q1_tfidf_weights = [weights.get(w, 0) for w in q1words.keys()]
    q2_tfidf_weights = [weights.get(w, 0) for w in q2words.keys()]
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights  = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    tfidfRatio = np.sum(shared_weights)*1.0 / np.sum(total_weights)
    q1_tfidf_sum     = sum(q1_tfidf_weights)
    q2_tfidf_sum     = sum(q2_tfidf_weights)
    q1_tfidf_mean    = np.mean(q1_tfidf_weights)
    q2_tfidf_mean    = np.mean(q2_tfidf_weights)
    q1_tfidf_min     = min(q1_tfidf_weights)
    q1_tfidf_max     = min(q1_tfidf_weights)
    q1_tfidf_range   = q1_tfidf_max - q1_tfidf_min
    q2_tfidf_min     = min(q2_tfidf_weights)
    q2_tfidf_max     = min(q2_tfidf_weights)
    q2_tfidf_range   = q2_tfidf_max - q2_tfidf_min
    return tfidfRatio, q1_tfidf_sum, q2_tfidf_sum, q1_tfidf_mean, q2_tfidf_mean, q1_tfidf_min, q1_tfidf_max, q1_tfidf_range, q2_tfidf_min, q2_tfidf_max, q2_tfidf_range

def sense2vec(q1p, q2p):   
    doc1 = nlp(q1p)
    doc2 = nlp(q2p)
    s2v1 = np.array([doc1[i]._.s2v_vec for i in range(len(doc1))])
    s2v1 = [i for i in s2v1 if i is not None]
    s2v1 = np.mean(s2v1)
    s2v2 = np.array([doc2[i]._.s2v_vec for i in range(len(doc2))])
    s2v2 = [i for i in s2v2 if i is not None]
    s2v2 = np.mean(s2v2)
    return np.dot(s2v1, s2v2.T)

def process_data(X,Y):
    examples = [j for j,x in enumerate(X) if not any(math.isnan(xs) for xs in x) and not any(math.isinf(xs) for xs in x) ]
    ff = []
    for i in examples:
      c = np.hstack((X[i],np.array([Y[i]])))
      ff.append(c)

    random.shuffle(ff)
    fy = [i[-1] for i in ff]
    fx = [i[:-1] for i in ff]
    return fx,fy

def create_features(q1,q2):
    feature_vector = []
    q1p = preprocessing_pipeline(q1)
    q2p = preprocessing_pipeline(q2)
    w1 = word_tokenize(q1p)
    w2 = word_tokenize(q2p)

    #Basic NLP features#
    feature_vector.extend(basic_nlp_features(q1,q2,w1,w2,q1p,q2p))

    #total number of words
    feature_vector.append(len(w1))
    feature_vector.append(len(w2))

    #total number of sentences
    feature_vector.append(len(sent_tokenize(q1p)))
    feature_vector.append(len(sent_tokenize(q2p)))

    #bag of words
    bow1, bow2= vectorizer_bow.transform([q1p,q2p])
    feature_vector.append(np.dot(bow1, bow2.T).toarray()[0][0])

    #uni and bi gram
    ngram1,ngram2 = vectorizer_ngram.transform([q1p, q2p])
    feature_vector.append(np.dot(ngram1, ngram2.T).toarray()[0][0])

    #Glove vector embeddings
    glove1 = np.mean(np.array([embeddings_dict[x] for x in w1 if x in embeddings_dict]))
    glove2 = np.mean(np.array([embeddings_dict[x] for x in w2 if x in embeddings_dict]))
    feature_vector.append(np.dot(glove1, glove2.T))

    #Google word2vec embeddings
    feature_vector.append(model.wmdistance(q1p, q2p))

    #Google norm word2vec embeddings
    feature_vector.append(norm_model.wmdistance(q1p, q2p))

    #named entity
    c,n = named_entity_features(q1p,q2p)
    feature_vector.extend([c,n])

    #Tri gram
    tgram1,tgram2 = vectorizer_3gram.transform([q1p, q2p])
    feature_vector.append(np.dot(tgram1, tgram2.T).toarray()[0][0])

    #IDF
    idf1, idf2 = vectorizer_idf.transform([q1p,q2p])
    feature_vector.append(np.dot(idf1, idf2.T).toarray()[0][0])

    #Spacy similarity
    feature_vector.append(nlp(q1p).similarity(nlp(q2p)))

    #Jaccard distance
    feature_vector.append(nltk.jaccard_distance(set(q1p),set(q2p)))

    #Word match score
    feature_vector.append(word_match_share(w1,w2))

    #1st word similarity score
    feature_vector.append(first_word_similarity(w1,w2))

    #tfidf word match scores
    feature_vector.extend(tfidf_word_match(w1,w2,q1p,q2p))

    #sense2vec embeddings
    feature_vector.append( sense2vec(q1p,q2p))

    return(feature_vector)


question_types = ["what", "how", "why", "is", "which", "can", "i", "who", "do", "where", "if", "does", "are", "when", "should", "will", "did", "has", "would", "have", "was", "could"]

#Loading Glove Embeddings

embeddings_dict = {}
with open("glove.6B.50d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        token = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[token] = vector

#Loading Word2Vec Embeddings

model = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/My Drive/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/My Drive/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)

#Loading Sense Embeddings

nlp = en_core_web_lg.load()
s2v = Sense2VecComponent(nlp.vocab).from_disk("/content/drive/My Drive/s2v_reddit_2019_lg")
nlp.add_pipe(s2v)

#Load training data

df = pd.read_csv('train_tsv.tsv', sep='\t', names=["is_duplicate", "question1", "question2", "id"])
df = df.set_index('id')
df = df[(df['question1'].isna() == False) & (df['question2'].isna() == False)]

training_data = list(df['question1'])+list(df['question2'])

training_data = [preprocessing_pipeline(i) for i in training_data]
tfidf_vectorizer = tfidf(training_data)
vectorizer_bow = bag_of_words(training_data, 0)
vectorizer_ngram = bag_of_words(training_data, 1)
vectorizer_3gram = trigram(training_data)
vectorizer_idf = idf(training_data)

#Preparing training data

train_X = []
train_y = []

for i in tqdm_notebook(range(len(df["question1"]))):
    train_X.append(create_features(df.iloc[i]["question1"], df.iloc[i]["question2"]))
    train_y.append(df.iloc[i]["is_duplicate"])

train_X, train_y = process_data(train_X,train_y)

#Load dev data

df1 = pd.read_csv('dev_tsv.tsv', sep='\t', names=["is_duplicate", "question1", "question2", "id"])
df1 = df1.set_index('id')
df1 = df1[(df1['question1'].isna() == False) & (df1['question2'].isna() == False)]

#Preparing dev data

dev_X = []
dev_y = []

for i in tqdm_notebook(range(len(df1["question1"]))):
    dev_X.append(create_features(df1.iloc[i]["question1"], df1.iloc[i]["question2"]))
    dev_y.append(df1.iloc[i]["is_duplicate"])

dev_X, dev_y = process_data(dev_X,dev_y)

#Load test data

df2 = pd.read_csv('test_tsv.tsv', sep='\t', names=["is_duplicate", "question1", "question2", "id"])
df2 = df2.set_index('id')
df2 = df2[(df2['question1'].isna() == False) & (df2['question2'].isna() == False)]

#Preparing test data

test_X = []
test_y = []

for i in tqdm_notebook(range(len(df2["question1"]))):
    test_X.append(create_features(df2.iloc[i]["question1"], df2.iloc[i]["question2"]))
    test_y.append(df2.iloc[i]["is_duplicate"])

test_X, test_y = process_data(test_X,test_y)