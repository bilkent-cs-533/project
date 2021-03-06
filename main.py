# Ignore DeprecationWarning caused by imp module
from warnings import catch_warnings, filterwarnings
with catch_warnings():
    filterwarnings("ignore", category=DeprecationWarning)
    import imp

from sys import getsizeof
from matplotlib import style
from TurkishStemmer import TurkishStemmer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from scipy import stats
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.sparse as sp 
from sklearn.preprocessing import StandardScaler

style.use("ggplot")

test_data_file_name = 'test_tweets.txt'
train_data_file_name = 'train_tweets.txt'
stop_words_file_name = 'stop_words_tr_147.txt'


def get_data_class_pairs(file_name):
    """ 
    input text data file name
    output text to class dictionary
    """
    data2class = dict()
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for l in lines:
            arr = l.split('\t')
            data2class[arr[0]] = arr[1]
    return data2class


def get_corpus(docs):
    """
    docs is list of list of words
    """
    d = dict()
    for doc in docs:
        for w in doc:
            d[w] = True

    i = 0
    for w in d:
        d[w] = i
        i += 1

    return d


def get_inverse_doc_freq(docs, corpus):
    """
    returns a dictionary of word to count of document which contains the word
    """
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


def get_char_gram_docs(docs, char_grams=[2]):
    """
    docs is list of list of words
    get character based ngram docs
    """
    ngram_docs = []
    for doc in docs:
        ngram_doc = []
        for word in doc:
            word = re.sub(r'[0-9\.\,\?]', '', word)
            for char_cnt in char_grams:
                ngram_doc += [word[i:i + char_cnt] for i in range(len(word) - char_cnt + 1)]
        ngram_docs.append(ngram_doc)
    return ngram_docs


# adds word grams to docs
def get_word_gram_docs(docs, word_grams=[2]):
  ngram_docs = []
  for doc in docs:
    ngram_doc = []
    for word_count in word_grams:
      ngram_doc += [' '.join(doc[i:i + word_count]) for i in range(len(doc) - word_count + 1)]
    ngram_docs.append(ngram_doc)
  return ngram_docs

@np.vectorize
def turkish_stemmer_vectorize(words):
    if len(words) == 0:
        return []
    stemmer = TurkishStemmer()
    return stemmer.stem(words)


def f5_stemmer(words):
    if len(words) == 0:
        return []
    return [w[:5] for w in words]


def get_feature_num_of_words_all_caps(docs):
    """
    docs is list of list of words
    x is feature matrix, np array
    this method adds a column to feature matrix which count number of words all caps
    """
    feature_col = np.zeros((len(docs), 1), dtype=np.uint8)
    for i,doc in enumerate(docs):
        feature_col[i] = sum([word.isupper() for word in doc])
    return feature_col

def get_feature_last_char_exclamation(docs):
    feature_col = np.zeros((len(docs), 1), dtype=np.uint8)
    for i,doc in enumerate(docs):
        feature_col[i] = sum([doc[-1].endswith('!')])
    return feature_col

def get_feature_num_of_elongated_words(docs):
    feature_col = np.zeros((len(docs), 1), dtype=np.uint8)
    regex = re.compile(r"(.)\1{3}")
    for i,doc in enumerate(docs):
        feature_col[i] = len([word for word in doc if regex.search(word)])
    return feature_col

def get_feature_pos_neg_word_count(docs):
    pos_words = {'iyi': True, 'süper': True, 'super': True, 'güzel': True, 'guzel': True, 'harika': True, 
    ';)': True, ':)': True, ':d': True, ':p': True, '=)': True }
    neg_words = {'kötü': True, 'berbat': True, 'çirkin': True, 'cirkin': True, 'iğrenç': True, 'igrenc': True, 
    ';(': True, ':(': True, '=(': True }
    
    feature_col = np.zeros((len(docs), 2), dtype=np.uint8)
    for i,doc in enumerate(docs):
        for pos_word in pos_words:
            feature_col[i, 0] += len([word for word in doc if pos_word in word ])
        for neg_word in neg_words:
            feature_col[i, 1] += len([word for word in doc if neg_word in word ])
    return feature_col


def get_features_as_freq_dist(docs, corpus, add_artificial_features=-1):
    """
    docs is list of list of words
    corpus is dictionary of word to index
    """
    l = np.zeros((len(docs), len(corpus)), dtype=np.uint8)
    for i, doc in enumerate(docs):
        d = dict()
        for word in doc:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
        for word in doc:
            l[i, corpus[word]] = d[word]

    return l


def get_features_merged(cleaned_docs, corpus, add_artificial_features=-1):
    binary_features = get_features_as_binary_freq_dist(cleaned_docs, corpus)
    binary_features = preprocessing.normalize((binary_features)).astype(np.float16)
    # scaler = StandardScaler()
    # scaler.fit(binary_features)
    # StandardScaler(copy=False, with_mean=True, with_std=True)
    freq_features = get_features_as_freq_dist(cleaned_docs, corpus)
    freq_features = preprocessing.normalize((freq_features)).astype(np.float16)

    if add_artificial_features != -1:
        if add_artificial_features == 0:
            artificial_features = get_feature_last_char_exclamation(cleaned_docs)
            artificial_features = preprocessing.normalize(artificial_features).astype(np.float16)
            return np.concatenate((binary_features, freq_features, artificial_features), axis=1)
        elif add_artificial_features == 1:
            artificial_features = get_feature_num_of_words_all_caps(cleaned_docs)
            artificial_features = preprocessing.normalize(artificial_features).astype(np.float16)
            return np.concatenate((binary_features, freq_features, artificial_features), axis=1)
        elif add_artificial_features == 2:
            artificial_features = get_feature_num_of_elongated_words(cleaned_docs)
            artificial_features = preprocessing.normalize(artificial_features).astype(np.float16)
            return np.concatenate((binary_features, freq_features, artificial_features), axis=1)
        elif add_artificial_features == 3:
            artificial_features = get_feature_pos_neg_word_count(cleaned_docs)
            artificial_features = preprocessing.normalize(artificial_features).astype(np.float16)
            return np.concatenate((binary_features, freq_features, artificial_features), axis=1)
        else:
            artificial_features1 = get_feature_last_char_exclamation(cleaned_docs)
            artificial_features1 = preprocessing.normalize(artificial_features1).astype(np.float16)
            artificial_features2 = get_feature_num_of_words_all_caps(cleaned_docs)
            artificial_features2 = preprocessing.normalize(artificial_features2).astype(np.float16)
            artificial_features3 = get_feature_num_of_elongated_words(cleaned_docs)
            artificial_features3 = preprocessing.normalize(artificial_features3).astype(np.float16)
            artificial_features4 = get_feature_pos_neg_word_count(cleaned_docs)
            artificial_features4 = preprocessing.normalize(artificial_features4).astype(np.float16)
            return np.concatenate((binary_features, freq_features, artificial_features1, artificial_features2, artificial_features3, artificial_features4), axis=1)

    else:
        return np.concatenate((binary_features, freq_features), axis=1)
        # return sp.hstack((binary_features, freq_features), format='csr')


def get_features_as_binary_freq_dist(docs, corpus, add_artificial_features=-1):
    """
    docs is list of list of words
    """
    l = np.zeros((len(docs), len(corpus)), dtype=np.uint8)
    for i, doc in enumerate(docs):
        for word in doc:
            l[i, corpus[word]] = 1

    return l


def get_features_tf_idf0(docs, corpus, add_artificial_features=-1):
    l = np.zeros((len(docs), len(corpus)))
    N = len(docs)
    idf_dict = get_inverse_doc_freq(docs, corpus)
    for i, doc in enumerate(docs):
        d = dict()
        for word in doc:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
        for word in doc:
            l[i, corpus[word]] = d[word] * math.log2(N / idf_dict[word])
            # l[i, corpus[word]] = 1 + math.log2(d[word])
            # l[i, corpus[word]] = (1 + math.log2(d[word])) * math.log2(N/idf_dict[word])
    return l


def get_features_tf_idf1(docs, corpus, add_artificial_features=-1):
    l = np.zeros((len(docs), len(corpus)))
    for i, doc in enumerate(docs):
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


def get_features_tf_idf2(docs, corpus, add_artificial_features=-1):
    l = np.zeros((len(docs), len(corpus)))
    N = len(docs)
    idf_dict = get_inverse_doc_freq(docs, corpus)
    for i, doc in enumerate(docs):
        d = dict()
        for word in doc:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
        for word in doc:
            # l[i, corpus[word]] = d[word] * math.log2(N/idf_dict[word])
            # l[i, corpus[word]] = 1 + math.log2(d[word])
            l[i, corpus[word]] = (1 + math.log2(d[word])) * \
                math.log2(N/idf_dict[word])
    return l


def get_cleaned_docs_from_file(file_name, stemmer=0, char_grams=[], word_grams=[]):
    datas = get_data_class_pairs(file_name)
    raw_docs = []
    cleaned_docs = []
    for d in datas.keys():
        raw_docs.append(d.split())

    for d in raw_docs:
        stop_words_removed = remove_stop_words(stop_words_file_name, d)
        if stemmer == 0:
            cleaned_docs.append(turkish_stemmer_vectorize(stop_words_removed))
        elif stemmer == 1:
            cleaned_docs.append(f5_stemmer(stop_words_removed))
        else:
            cleaned_docs.append(stop_words_removed)
    if len(char_grams) > 0:
        cleaned_docs = get_char_gram_docs(cleaned_docs, char_grams)

    if len(word_grams) > 0:
        cleaned_docs = get_word_gram_docs(cleaned_docs, word_grams)
    
    return cleaned_docs, np.array(list(datas.values()))


def model_runner(clf, features, features2, kfold=False):
    global cleaned_docs, y, cleaned_docs2, y2, corpus
    
    if kfold:
        kf = KFold(n_splits=10)
        succ = []
        for train_index, test_index in kf.split(cleaned_docs):
            clf.fit(features[train_index], y[train_index])
            predictions = clf.predict(features[test_index])
            succ.append(sum(predictions == y[test_index]) / len(predictions))
        # print(succ)
        return succ
    else:
        clf.fit(np.array(features), y)
        predictions = clf.predict(np.array(features2))
        succ = sum(predictions == y2) / len(predictions)
        # print(succ)
        return [succ]


def run_svc_for_func(train_features, test_features, kfold=False):
    """
    generates classifier and call model runner with feature generator function
    """
    clf = LinearSVC(random_state=0, tol=1e-5)
    return model_runner(clf, train_features, test_features, kfold)


def run_sgd_for_func(train_features, test_features, kfold=False):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    return model_runner(clf, train_features, test_features, kfold)


def run_decision_tree_for_func(train_features, test_features, kfold=False):
    clf = tree.DecisionTreeClassifier()
    return model_runner(clf, train_features, test_features, kfold)


def run_random_forest_for_func(train_features, test_features, kfold=False):
    clf = RandomForestClassifier(n_estimators=10)
    return model_runner(clf, train_features, test_features, kfold)


def run_k_means_for_func(train_features, test_features, kfold=False):
    clf = KMeans(init='k-means++', n_clusters=3, n_init=10)
    return model_runner(clf, train_features, test_features, kfold)


def run_mlp_for_func(train_features, test_features, kfold=False):
    clf = clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ), random_state=1)
    return model_runner(clf, train_features, test_features, kfold)


def experiment_runner(model_func, train_features, test_features=None, kfold=False):
    start = time.time()
    accuracies = model_func(train_features, test_features, kfold)
    end = time.time()
    print('Executed in ', end - start, ' secs')
    return accuracies


def statistically_different(data1, data2, alpha=0.05):
    """
    alpha is the confidence level. It is initially 0.05 which means %95 confidence. 
    returns True if data1 and data2 statistically different
    """
    t_value, p_value = stats.ttest_rel(data1, data2)
    crititcal_t_value = stats.t.ppf(1 - (alpha / 2), len(data1))
    print("ttest parameters:", t_value, p_value, crititcal_t_value)

    if t_value > crititcal_t_value:
        return True

    return False


def dump_result(filename, result_list):
    """
    writes each item to a line
    overwrites old file
    """
    text = "\n".join(str(x) for x in result_list)
    with open(".\\results\\" + filename, "w") as f:
        f.write(text)

def kfold_experiment():
    global method_dict, feature_dict, cleaned_docs, corpus
    # 10-Fold experiments for all models
    for feature_type, feature_generator_func in feature_dict.items():
        # generating features once 
        print("Extracting", feature_type)
        start = time.time()
        features = feature_generator_func(cleaned_docs, corpus)
        end = time.time()
        print('Executed in ', end - start, ' secs')
        print(feature_type, " size :", getsizeof(features)/(1024**2), "MB")

        for method_name, model_func in method_dict.items():
            print("Running", method_name, "with",  feature_type)
            acc_list = experiment_runner(model_func, features, kfold = True)
            result_filename = method_name + "_" + feature_type + ".txt"
            dump_result(result_filename, acc_list)

def test_experiment(model_func, feature_generator_func, dump_filename="test_dump.txt", add_artificial_features=-1, kfold=False):
    global cleaned_docs, cleaned_docs2, corpus
    print("Test is starting.")
    
    print("Extracting train and test features...")
    start = time.time()
    train_features = feature_generator_func(cleaned_docs, corpus, add_artificial_features)
    test_features = feature_generator_func(cleaned_docs2, corpus, add_artificial_features)
    end = time.time()
    print('Executed in ', end - start, ' secs')
    
    print("Running model...")
    acc_list = experiment_runner(model_func, train_features, test_features,kfold)
    dump_result(dump_filename, acc_list)

method_dict = {
    "SVC": run_svc_for_func,
    "SGD": run_sgd_for_func,
    "DT": run_decision_tree_for_func,
    "RF": run_random_forest_for_func,
    "K_MEANS": run_k_means_for_func,
    "MLP": run_mlp_for_func
}
print("Preprocessing...")
start = time.time()

# cleaned_docs, y = get_cleaned_docs_from_file(train_data_file_name, stemmer=-1, char_grams=[1,2,3,4], word_grams=[])
# cleaned_docs2, y2 = get_cleaned_docs_from_file(test_data_file_name, stemmer=-1, char_grams=[1,2,3,4], word_grams=[])
cleaned_docs, y = get_cleaned_docs_from_file(train_data_file_name)
cleaned_docs2, y2 = get_cleaned_docs_from_file(test_data_file_name)
corpus = get_corpus(cleaned_docs + cleaned_docs2)

end = time.time()
print('Executed in ', end - start, ' secs')

feature_dict = {
    "binary_features": get_features_as_binary_freq_dist,
    "freq_features": get_features_as_freq_dist,
    "merged_features": get_features_merged
}

# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_last_char_exclamation.txt", 0, True)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_num_of_all_word_cap.txt", 1, True)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_num_of_elongated_word.txt", 2, True)
test_experiment(run_svc_for_func, get_features_tf_idf0, "SVC_tfidf0_turkish_stemmer.txt", -1, True)
test_experiment(run_svc_for_func, get_features_tf_idf1, "SVC_tfidf1_turkish_stemmer.txt", -1, True)
test_experiment(run_svc_for_func, get_features_tf_idf2, "SVC_tfidf2_turkish_stemmer.txt", -1, True)


# kfold_experiment()
# test_experiment(run_svc_for_func, get_features_as_binary_freq_dist, "SVC_binary_test_dump.txt")
# test_experiment(run_svc_for_func, get_features_merged, "SVC_kfold_merged_features_2_3_4_char_grams_dump.txt", False, True)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_2_3_char_grams_dump.txt")
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_2_3_char_grams_merged_doc_representations_dump.txt")
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_2_3_char_grams_last_char_exclamation_dump.txt", True)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_2_3_char_grams_remove_num_dot_dump.txt", False)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_2_3_char_grams_with_artifical_features_dump.txt", True)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_dump.txt", False)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_char_word_gram_doc_representation_dump.txt", False)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_2_word_grams_dump.txt", False)

# test_experiment(run_svc_for_func, get_features_merged, "SVC_2_3_char_gram_no_stemmer.txt", False, True)

# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_with_num_of_all_caps_word_dump.txt", True)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_with_last_char_exclamation_dump.txt", True)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_with_elongated_word_count_dump.txt", True)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_lower_docs_dump.txt", False)
# test_experiment(run_svc_for_func, get_features_merged, "SVC_merged_features_with_pos_neg_word_count_dump.txt", True)

# print("Statistically different:", statistically_different(acc_list2, acc_list1))
