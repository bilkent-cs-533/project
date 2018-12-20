# Ignore DeprecationWarning caused by imp module
from warnings import catch_warnings, filterwarnings
with catch_warnings():
    filterwarnings("ignore", category=DeprecationWarning)
    import imp

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


@np.vectorize
def turkish_stemmer_vectorize(words):
    if len(words) == 0:
        return []
    stemmer = TurkishStemmer()
    return stemmer.stem(words)


def get_features_as_freq_dist(docs, corpus):
    """
    docs is list of list of words
    corpus is dictionary of word to index
    """
    l = np.zeros((len(docs), len(corpus)))
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


def get_features_merged(docs, corpus):
    f1 = get_features_as_freq_dist(docs, corpus)
    f1 = preprocessing.normalize(f1)
    f2 = get_features_as_binary_freq_dist(docs, corpus)
    f2 = preprocessing.normalize(f2)

    return np.concatenate((f1, f2), axis=1)


def get_features_as_binary_freq_dist(docs, corpus):
    """
    docs is list of list of words
    """
    l = np.zeros((len(docs), len(corpus)))
    for i, doc in enumerate(docs):
        for word in doc:
            l[i, corpus[word]] = 1

    return l


def get_features_tf_idf0(docs, corpus):
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


def get_features_tf_idf1(docs, corpus):
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


def get_features_tf_idf2(docs, corpus):
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


def model_runner(clf, feature_generator_func, kfold=False):
    cleaned_docs, y = get_cleaned_docs_from_file(train_data_file_name)
    cleaned_docs2, y2 = get_cleaned_docs_from_file(test_data_file_name)
    corpus = get_corpus(cleaned_docs + cleaned_docs2)
    features = feature_generator_func(cleaned_docs, corpus)
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
        features2 = feature_generator_func(cleaned_docs2, corpus)
        clf.fit(features, y)
        predictions = clf.predict(features2)
        succ = sum(predictions == y2) / len(predictions)
        # print(succ)
        return succ


def run_svc_for_func(feature_generator_func, kfold=False):
    """
    generates classifier and call model runner with feature generator function
    """
    clf = LinearSVC(random_state=0, tol=1e-5)
    return model_runner(clf, feature_generator_func, kfold)


def run_sgd_for_func(feature_generator_func, kfold=False):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    return model_runner(clf, feature_generator_func, kfold)


def run_decision_tree_for_func(feature_generator_func, kfold=False):
    clf = tree.DecisionTreeClassifier()
    return model_runner(clf, feature_generator_func, kfold)


def run_random_forest_for_func(feature_generator_func, kfold=False):
    clf = RandomForestClassifier(n_estimators=10)
    return model_runner(clf, feature_generator_func, kfold)


def run_k_means_for_func(feature_generator_func, kfold=False):
    clf = KMeans(init='k-means++', n_clusters=3, n_init=10)
    return model_runner(clf, feature_generator_func, kfold)


def run_mlp_for_func(feature_generator_func, kfold=False):
    clf = clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ), random_state=1)
    return model_runner(clf, feature_generator_func, kfold)


def experiment_runner(model_func, feature_generator_func, kfold=False):
    start = time.time()
    accuracies = model_func(feature_generator_func, kfold)
    end = time.time()
    print('executed in ', end - start, ' secs')
    return accuracies


def statistically_different(data1, data2, alpha=0.05):
    """
    alpha is the confidence level. It is initially 0.05 which means %95 confidence. 
    returns True if data1 and data1 statistically different
    """
    t_value, p_value = stats.ttest_rel(data1, data2)
    crititcal_t_value = stats.t.ppf(1 - (alpha / 2), len(data1))
    print("ttest paramters:", t_value, p_value, crititcal_t_value)

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


# SVC
acc_list1 = experiment_runner(run_svc_for_func, get_features_as_freq_dist, True)
dump_result("SVC_freq_features.txt", acc_list1)

acc_list2 = experiment_runner(run_svc_for_func, get_features_as_binary_freq_dist, True)
dump_result("SVC_binary_features.txt", acc_list2)

acc_list3 = experiment_runner(run_svc_for_func, get_features_merged, True)
dump_result("SVC_merged_features.txt", acc_list3)

# SGD
acc_list4 = experiment_runner(run_sgd_for_func, get_features_as_freq_dist, True)
dump_result("SGD_freq_features.txt", acc_list4)

acc_list5 = experiment_runner(run_sgd_for_func, get_features_as_binary_freq_dist, True)
dump_result("SGD_binary_features.txt", acc_list5)

acc_list6 = experiment_runner(run_sgd_for_func, get_features_merged, True)
dump_result("SGD_merged_features.txt", acc_list6)

# experiment_runner(run_decision_tree_for_func, get_features_merged)
# experiment_runner(run_random_forest_for_func, get_features_merged)
# experiment_runner(run_k_means_for_func, get_features_merged)
# experiment_runner(run_svc_for_func, get_features_tf_idf0)
# experiment_runner(run_mlp_for_func, get_features_as_binary_freq_dist)
# experiment_runner(run_mlp_for_func, get_features_tf_idf0)

# print("Statistically different:", statistically_different(acc_list2, acc_list1))
