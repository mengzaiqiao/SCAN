import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(Fn, adj, Fa, features_orig, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['Fn']: Fn})
    feed_dict.update({placeholders['features_orig']: features_orig})
    feed_dict.update({placeholders['Fa']: Fa})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    adj_row = adj.nonzero()[0]
    adj_col = adj.nonzero()[1]
    
    #get deges from adjacant matrix
    edges = []
    edges_dic = {}
    for i in range(len(adj_row)):
        edges.append([adj_row[i], adj_col[i]])
        edges_dic[(adj_row[i], adj_col[i])] = 1
    
    #split the dataset into training,validation and test dataset
    false_edges_dic = {}
    num_test = int(np.floor(len(edges) / 10.))
    num_val = int(np.floor(len(edges) / 20.))
    all_edge_idx = np.arange(len(edges))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    edges = np.array(edges)
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    
    
    test_edges_false = []
    val_edges_false = []
    while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if (i, j) in edges_dic:
            continue
        if (j, i) in edges_dic:
            continue
        if (i, j) in false_edges_dic:
            continue
        if (j, i) in false_edges_dic:
            continue
        else:
            false_edges_dic[(i, j)] = 1
            false_edges_dic[(j, i)] = 1
        if np.random.random_sample() > 0.333 :
            if len(test_edges_false) < num_test :
                test_edges_false.append((i, j))
            else:
                if len(val_edges_false) < num_val :
                    val_edges_false.append([i, j])
        else:
            if len(val_edges_false) < num_val :
                val_edges_false.append([i, j])
            else:
                if len(test_edges_false) < num_test :
                    test_edges_false.append([i, j])
    
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_feas(features):
    fea_row = features.nonzero()[0]
    fea_col = features.nonzero()[1]
    
    feas = []
    feas_dic = {}
    for i in range(len(fea_row)):
        feas.append([fea_row[i], fea_col[i]])
        feas_dic[(fea_row[i], fea_col[i])] = 1
        
    false_feas_dic = {}
    num_test = int(np.floor(len(feas) / 10.))
    num_val = int(np.floor(len(feas) / 20.))
    all_fea_idx = np.arange(len(feas))
    np.random.shuffle(all_fea_idx)
    val_fea_idx = all_fea_idx[:num_val]
    test_fea_idx = all_fea_idx[num_val:(num_val + num_test)]
    feas = np.array(feas)
    test_feas = feas[test_fea_idx]
    val_feas = feas[val_fea_idx]
    train_feas = np.delete(feas, np.hstack([test_fea_idx, val_fea_idx]), axis=0)
    
    
    test_feas_false = []
    val_feas_false = []
    while len(test_feas_false) < num_test or len(val_feas_false) < num_val:
        i = np.random.randint(0, features.shape[0])
        j = np.random.randint(0, features.shape[1])
        if (i, j) in feas_dic:
            continue
        if (i, j) in false_feas_dic:
            continue
        else:
            false_feas_dic[(i, j)] = 1
        if np.random.random_sample() > 0.333 :
            if len(test_feas_false) < num_test :
                test_feas_false.append([i, j])
            else:
                if len(val_feas_false) < num_val :
                    val_feas_false.append([i, j])
        else:
            if len(val_feas_false) < num_val :
                val_feas_false.append([i, j])
            else:
                if len(test_feas_false) < num_test :
                    test_feas_false.append([i, j])
    data = np.ones(train_feas.shape[0])
    fea_train = sp.csr_matrix((data, (train_feas[:, 0], train_feas[:, 1])), shape=features.shape)
    return fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false


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

def get_labels(file,ratio=0.1):
    label_file = "data/" + file + ".label"
    full_labels = read_label(label_file)
    
    #get how many kinds of labels in the data
    num_labels_kind = len(np.unique(full_labels))
    
    # to get the index of labels to be remained
    num_labels = len(full_labels)
    idx = np.arange(num_labels)
    np.random.shuffle(idx)
    labels_keep = idx[:int(num_labels*ratio)]
    labels_test = idx[-1000:]
    
    #get the position of nodes whose labels have been remained
    labels_pos = np.array([False]*num_labels)
    labels_pos[labels_keep] = True
    labels_test_pos = np.array([False]*num_labels)
    labels_test_pos[labels_test] = True
    
    #get the labels remained and labels unremained
    labels = full_labels[labels_pos]
    #unlabels = full_labels[labels_pos==False]
    
    return labels_pos,labels_test_pos,labels,full_labels,num_labels_kind


def labels_onehot(labels_pos,labels,num_labels):
    """
    get encoding for y: one-hot encoding for label data, and 0 to denote unlabel data
    """
    #labels starting from 1, use 0 to represent nodes without labels
    all_labels = np.zeros(shape = (len(labels_pos)))
    all_labels[labels_pos] = labels
    all_labels_onehot =  tf.one_hot(all_labels,num_labels+1)[:,1:]
    
    return all_labels_onehot
    
    