import numpy as np
from TurkishStemmer import TurkishStemmer 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time
import math

test_data_file_name = 'test_tweets.txt'
train_data_file_name = 'train_tweets.txt'
stop_words_file_name = 'stop_words_tr_147.txt'

# input text data file name
# output text to class dictionary
def get_data_class_pairs(file_name):
  data2class = dict()
  with open(file_name, 'r', encoding='utf-8') as f:
    for l in f.readlines():
      arr = l.split('\t')
      data2class[arr[0]] = arr[1]
  return data2class

# docs is list of list of words
def get_corpus(docs):
  d = dict()
  for doc in docs:
    for w in doc:
      d[w] = True
  
  i = 0
  for w in d:
    d[w] = i
    i += 1

  return d

# returns a dictionary of word to count of document which contains the word
def get_inverse_doc_freq(docs, corpus):
  d = dict()
  for word in corpus:
    i = 0
    for doc in docs:
      if word in doc:
        i += 1
    d[word] = i

  return d


def remove_stop_words(stop_words_file_name, words):
  with open(stop_words_file_name, 'r', encoding="utf-8") as myfile:
    stop_words = myfile.read().lower().strip().split()

  return [x for x in words if x not in stop_words]

@np.vectorize
def turkish_stemmer_vectorize(words):
  if len(words) == 0:
    return []
  stemmer = TurkishStemmer()
  return stemmer.stem(words)

# docs is list of list of words
# corpus is dictionary of word to index
def get_features_as_freq_dist(docs, corpus):
  l = np.zeros((len(docs), len(corpus)))
  for i,doc in enumerate(docs):
    d = dict()
    for word in doc:
      if word in d:
        d[word] += 1
      else:
        d[word] = 1
    for word in doc:
      l[i, corpus[word]] = d[word]
      
  return l

def get_features_merged(docs, corpus):
  f1 = get_features_as_freq_dist(docs, corpus)
  f1 = preprocessing.normalize(f1)
  f2 = get_features_as_binary_freq_dist(docs, corpus)
  f2 = preprocessing.normalize(f2)

  return np.concatenate((f1,f2), axis=1)

# docs is list of list of words
# docs is list of list of words
def get_features_as_binary_freq_dist(docs, corpus):
  l = np.zeros((len(docs), len(corpus)))
  for i,doc in enumerate(docs):
    for word in doc:
      l[i, corpus[word]] = 1
      
  return l

def get_features_tf_idf0(docs, corpus):
  l = np.zeros((len(docs), len(corpus)))
  N = len(docs)
  idf_dict = get_inverse_doc_freq(docs, corpus)
  for i,doc in enumerate(docs):
    d = dict()
    for word in doc:
      if word in d:
        d[word] += 1
      else:
        d[word] = 1
    for word in doc:
      l[i, corpus[word]] = d[word] * math.log2(N/idf_dict[word])
      # l[i, corpus[word]] = 1 + math.log2(d[word])
      # l[i, corpus[word]] = (1 + math.log2(d[word])) * math.log2(N/idf_dict[word])
  return l

def get_features_tf_idf1(docs, corpus):
  l = np.zeros((len(docs), len(corpus)))
  for i,doc in enumerate(docs):
    d = dict()
    for word in doc:
      if word in d:
        d[word] += 1
      else:
        d[word] = 1
    for word in doc:
      # l[i, corpus[word]] = d[word] * math.log2(N/idf_dict[word])
      l[i, corpus[word]] = 1 + math.log2(d[word])
      # l[i, corpus[word]] = (1 + math.log2(d[word])) * math.log2(N/idf_dict[word])
  return l

def get_features_tf_idf2(docs, corpus):
  l = np.zeros((len(docs), len(corpus)))
  N = len(docs)
  idf_dict = get_inverse_doc_freq(docs, corpus)
  for i,doc in enumerate(docs):
    d = dict()
    for word in doc:
      if word in d:
        d[word] += 1
      else:
        d[word] = 1
    for word in doc:
      # l[i, corpus[word]] = d[word] * math.log2(N/idf_dict[word])
      # l[i, corpus[word]] = 1 + math.log2(d[word])
      l[i, corpus[word]] = (1 + math.log2(d[word])) * math.log2(N/idf_dict[word])
  return l

def get_cleaned_docs_from_file(file_name):
  datas = get_data_class_pairs(file_name)
  raw_docs = []
  cleaned_docs = []
  for d in datas.keys():
    raw_docs.append(d.split())
  for d in raw_docs:
    stop_words_removed = remove_stop_words(stop_words_file_name, d)
    cleaned_docs.append(turkish_stemmer_vectorize(stop_words_removed))

  return cleaned_docs, np.array(list(datas.values()))

def model_runner(clf, feature_generator_func):
  cleaned_docs, y = get_cleaned_docs_from_file(train_data_file_name)
  cleaned_docs2, y2 = get_cleaned_docs_from_file(test_data_file_name)
  corpus = get_corpus(cleaned_docs + cleaned_docs2)
  features = feature_generator_func(cleaned_docs, corpus)
  features2 = feature_generator_func(cleaned_docs2, corpus)
  clf.fit(features, y)
  predictions = clf.predict(features2)
  succ = sum(predictions==y2) / len(predictions)
  print(succ)
  return succ

# %%time
# generates classifier and call model runner with feature generator function
def run_svc_for_func(feature_generator_func):
  clf = LinearSVC(random_state=0, tol=1e-5)
  model_runner(clf, feature_generator_func)

def run_sgd_for_func(feature_generator_func):
  clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
  model_runner(clf, feature_generator_func)

def run_decision_tree_for_func(feature_generator_func):
  clf = tree.DecisionTreeClassifier()
  model_runner(clf, feature_generator_func)

def run_random_forest_for_func(feature_generator_func):
  clf = RandomForestClassifier(n_estimators=10)
  model_runner(clf, feature_generator_func)

def run_k_means_for_func(feature_generator_func):
  clf = KMeans(init='k-means++', n_clusters=3, n_init=10)
  model_runner(clf, feature_generator_func)

def experiment_runner(model_func, feature_generator_func):
  start = time.time()
  run_svc_for_func(get_features_as_freq_dist)
  end = time.time()
  print( 'executed in ', end - start, ' secs')

# experiment_runner(run_svc_for_func, get_features_as_freq_dist)
# experiment_runner(run_svc_for_func, get_features_as_binary_freq_dist)
# experiment_runner(run_svc_for_func, get_features_merged)
# experiment_runner(run_sgd_for_func, get_features_merged)
# experiment_runner(run_decision_tree_for_func, get_features_merged)
# experiment_runner(run_random_forest_for_func, get_features_merged)
# experiment_runner(run_k_means_for_func, get_features_merged)
experiment_runner(run_svc_for_func, get_features_tf_idf0)

