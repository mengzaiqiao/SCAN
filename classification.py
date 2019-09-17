from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.sparse as sp
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score,accuracy_score


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_labels(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, y, ty, ally = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = sp.lil_matrix((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    labels = sp.vstack((ally, ty)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    print("load_data finished!")
    row, col = labels.nonzero()
    label_real = []
    for j in col:
        label_real.append(j + 1)
    return  label_real


def multiclass_node_classification_eval(X, y, ratio=0.5, rnd=2018):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=rnd)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    accuracy = accuracy_score(y_test,y_pred,normalize=True)

    return macro_f1, micro_f1,accuracy


def node_classification(Embeddings, y, ratio):
    macro_f1_avg = 0
    micro_f1_avg = 0
    for i in range(10):
        rnd = np.random.randint(2018)
        macro_f1, micro_f1,accuracy = multiclass_node_classification_eval(
            Embeddings, y, ratio, rnd)
        macro_f1_avg += macro_f1
        micro_f1_avg += micro_f1
    macro_f1_avg /= 10
    micro_f1_avg /= 10

    return macro_f1_avg,micro_f1_avg,accuracy
    
def read_label(inputFileName):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()
    N = len(lines)
    y = np.zeros(N, dtype=int)
    i = 0
    for line in lines:
        l = line.strip("\n\r")
        y[i] = int(l)
        i += 1
    return y


def embedding_classifier(datasetname,ratio=0.2):
    #''cora','pubmed','Flickr', 'BlogCatalog'
    print('dataset:', datasetname, ',ratio:', ratio)
    embedding_node_result_file = "result/AGAE_{}_n_mu.emb.npy".format(datasetname)
    label_file = "data/" + datasetname + ".label"
    y = read_label(label_file)
    # print(len(y))
    Embeddings = np.load(embedding_node_result_file)
    # print(Embeddings.shape)
    macro_f1_avg,micro_f1_avg,accuracy = node_classification(Embeddings, y, ratio)
    return macro_f1_avg,micro_f1_avg,accuracy
        

