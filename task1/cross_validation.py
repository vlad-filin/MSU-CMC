import numpy as np
from sklearn.neighbors import NearestNeighbors
import skimage.transform as skt
import scipy.ndimage.interpolation as ndimage
import scipy.ndimage.filters as filt
import nearest_neighbors as My_KNN


#   Для прохождения Я-контеста был полностью вставлен код nearest.neighbors.py


class KNNClassifier:
    def __init__(self, k=5, strategy='my_own', metric='euclidean', weights=False,
                 test_block_size=1000):
        self.strategy = strategy
        self.weights = weights
        self.test_block_size = test_block_size
        self.metric = metric
        self.k = k
        if strategy != 'my_own':
            self.classifier = NearestNeighbors(k, algorithm=strategy, metric=metric)
            
    def fit(self, X, y):
        self.train_y = y
        if self.strategy == 'my_own':
            self.train_X = X
        else:
            self.classifier.fit(X, y)
        self.Nclass = np.unique(y)
    
    def euclid_dist(X, Y):
        return ((X ** 2).sum(axis=1)[:, np.newaxis] + (Y ** 2).sum(axis=1)[np.newaxis, :] -
                2 * X.dot(Y.transpose())) ** (1 / 2)
    
    def cosine_dist(X, Y):
        return (np.ones((X.shape[0], Y.shape[0])) - X.dot(Y.transpose()) /
                (((X ** 2).sum(axis=1)[:, np.newaxis] ** (1/2)) * 
                 ((Y ** 2).sum(axis=1)[np.newaxis, :] ** (1/2))))
        
    def find_kneighbors(self, X, return_distance=True):
        indexex = np.zeros((0, self.k))
        distance = np.zeros((0, self.k))
        tbl = self.test_block_size
        if self.strategy == 'my_own':
            if self.metric == 'euclidean':
                metric_func = KNNClassifier.euclid_dist
            else:
                metric_func = KNNClassifier.cosine_dist
            for i in range(X.shape[0] // tbl):
                P = metric_func(X[i * tbl:(i + 1) * tbl, :], self.train_X)
                ind = np.argpartition(P, range(self.k), axis=1)[:, 0:self.k]
                indexex = np.vstack((indexex, ind))
                if return_distance is True:
                    dist = np.partition(P, range(self.k), axis=1)[:, 0:self.k]
                    distance = np.vstack((distance, dist))
            P = metric_func(X[(X.shape[0] // tbl) * tbl:], self.train_X)
            ind = np.argpartition(P, range(self.k), axis=1)[:, 0:self.k]
            indexex = np.vstack((indexex, ind))
            if return_distance is True:
                dist = np.partition(P, range(self.k), axis=1)[:, 0:self.k]
                distance = np.vstack((distance, dist))           
        else:
            for i in range(X.shape[0] // self.test_block_size):
                dist, ind = self.classifier.kneighbors(X[i * tbl:(i + 1) * tbl, :],
                                                       n_neighbors=self.k, return_distance=True)
                distance = np.vstack((distance, dist))
                indexex = np.vstack((indexex, ind))
            if X.shape[0] % tbl != 0:
                dist, ind = self.classifier.kneighbors(X[(X.shape[0] // tbl) * tbl:, :],
                                                       n_neighbors=self.k, return_distance=True)
                distance = np.vstack((distance, dist))
                indexex = np.vstack((indexex, ind))
        if return_distance is True:
            return (distance, indexex.astype(int))
        else:
            return indexex.astype(int)
        
    def predict(self, X):
        answer = np.zeros(X.shape[0])
        if self.weights is False:
            knn = self.find_kneighbors(X, return_distance=False)
            classes = self.train_y[knn]
            m = np.zeros(X.shape[0])
            for i in self.Nclass:
                current = (classes == i).sum(axis=1)
                answer[(current > m)] = i
                m[(current - m) > 0] = current[(current - m) > 0]          
        else:
            knn = self.find_kneighbors(X, return_distance=True)
            weights = (knn[0] + 10 ** (-5)) ** (-1)
            classes = self.train_y[knn[1]]
            m = np.zeros(X.shape[0])
            for i in self.Nclass:
                current = ((classes == i) * weights).sum(axis=1)
                answer[(current > m)] = i
                m[(current - m) > 0] = current[(current - m) > 0]
        return answer    
    
    def predict_k(self, classes, knn_dist=None):
        answer = np.ones(classes.shape[0])
        if knn_dist is None:
            weights = np.ones(classes.shape)
        else:
            weights = (knn_dist + 10 ** (-5)) ** (-1)
        m = np.zeros(classes.shape[0])
        for i in self.Nclass:
            current = ((classes == i) * weights).sum(axis=1)
            answer[(current > m)] = i
            m[(current - m) > 0] = current[(current - m) > 0] 
        return answer    


def kfold(n, n_folds):
    tmp = np.arange(n)
    np.random.shuffle(tmp)
    tmp2 = np.arange(n)
    x = np.array_split(tmp, n_folds)
    y = []
    tmp1 = np.zeros((n,))
    for i in x:
        tmp1 = np.ones((n,))*2.5
        tmp1[i] = i
        y.append(tmp2[tmp1 != tmp2])
    return list(zip(y, x))


def acc_scorer(y_true, y_pred):
    return (y_true == y_pred).sum() / y_pred.shape[0]
    
    
def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    if score == 'accuracy':
        scorer = acc_scorer
    else:
        scorer = score
    if cv is None:
        cv = kfold(X.shape[0], 3)
    max_k = max(k_list)
    cl = KNNClassifier(k=max_k, **kwargs)
    acc = []
    for i in range(len(cv)):
        cl.fit(X[cv[i][0], :], y[(cv[i][0])])
        knn_ind = cl.find_kneighbors(X[cv[i][1]], return_distance=cl.weights)
        knn_dist = None
        if cl.weights is True:
            knn_dist = knn_ind[0]
            knn_ind = knn_ind[1]
        for k in range(len(k_list)):
            if i == 0:
                if knn_dist is None:
                    acc.append([scorer(y[cv[i][1]],
                                       cl.predict_k(y[cv[i][0]][knn_ind[:, 0:k_list[k]]]))])
                else:
                    acc.append([scorer(y[cv[i][1]],
                                       cl.predict_k(y[cv[i][0]][knn_ind[:, 0:k_list[k]]],
                                                    knn_dist[:, 0:k_list[k]]))])
            else:
                if knn_dist is None:
                    acc[k].append(scorer(y[cv[i][1]],
                                         cl.predict_k(y[cv[i][0]][knn_ind[:, 0:k_list[k]]])))
                else:
                    acc[k].append(scorer(y[cv[i][1]],
                                         cl.predict_k(y[cv[i][0]][knn_ind[:, 0:k_list[k]]],
                                                      knn_dist[:, 0:k_list[k]])))
            
    return dict(zip(k_list, acc))


def knn_cross_val_score_for5(X, y, param, cv, n_folds=3, change='rotate', **kwargs):
    scorer = acc_scorer
    cl = KNNClassifier(**kwargs)
    acc = []
    if change == 'rotate':
        for i in range(len(cv)):
            cl.fit(X[cv[i][0], :], y[(cv[i][0])])
            knn_dist, knn_ind = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
            for p in [param, -param]:
                X_new = np.array([skt.rotate(im.reshape(28, 28), p).ravel()
                                  for im in X[cv[i][0], :]])
                cl.fit(X_new, y[(cv[i][0])])
                knn_dist_tmp, knn_ind_tmp = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
                knn_ind[knn_dist > knn_dist_tmp] = knn_ind_tmp[knn_dist > knn_dist_tmp]
                knn_dist[knn_dist > knn_dist_tmp] = knn_dist_tmp[knn_dist > knn_dist_tmp]
                del(X_new)
                del(knn_ind_tmp)
                del(knn_dist_tmp)
            if cl.weights is True:
                acc.append(scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind])))
            else:
                acc.append([scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind], knn_dist))])
        return acc
    if change == 'shift':
        for i in range(len(cv)):
            cl.fit(X[cv[i][0], :], y[(cv[i][0])])
            knn_dist, knn_ind = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
            for p in [param, -param]:
                X_new = np.array([ndimage.shift(im.reshape(28, 28), [p, 0]).ravel()
                                  for im in X[cv[i][0], :]])
                cl.fit(X_new, y[(cv[i][0])])
                knn_dist_tmp, knn_ind_tmp = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
                knn_ind[knn_dist > knn_dist_tmp] = knn_ind_tmp[knn_dist > knn_dist_tmp]
                knn_dist[knn_dist > knn_dist_tmp] = knn_dist_tmp[knn_dist > knn_dist_tmp]
                del(X_new)
                del(knn_ind_tmp)
                del(knn_dist_tmp)
            for p in [param, -param]:
                X_new = np.array([ndimage.shift(im.reshape(28, 28), [0, p]).ravel()
                                  for im in X[cv[i][0], :]])
                cl.fit(X_new, y[(cv[i][0])])
                knn_dist_tmp, knn_ind_tmp = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
                knn_ind[knn_dist > knn_dist_tmp] = knn_ind_tmp[knn_dist > knn_dist_tmp]
                knn_dist[knn_dist > knn_dist_tmp] = knn_dist_tmp[knn_dist > knn_dist_tmp]
                del(X_new)
                del(knn_ind_tmp)
                del(knn_dist_tmp)
            if cl.weights is True:
                acc.append(scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind])))
            else:
                acc.append([scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind], knn_dist))])
        return acc
    if change == 'filter':
        for i in range(len(cv)):
            cl.fit(X[cv[i][0], :], y[(cv[i][0])])
            knn_dist, knn_ind = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
            p = param ** (1/2)
            X_new = np.array([filt.gaussian_filter(im.reshape(28, 28), p).ravel() 
                              for im in X[cv[i][0], :]])
            cl.fit(X_new, y[(cv[i][0])])
            knn_dist_tmp, knn_ind_tmp = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
            knn_ind[knn_dist > knn_dist_tmp] = knn_ind_tmp[knn_dist > knn_dist_tmp]
            knn_dist[knn_dist > knn_dist_tmp] = knn_dist_tmp[knn_dist > knn_dist_tmp]
            del(X_new)
            del(knn_ind_tmp)
            del(knn_dist_tmp)
            if cl.weights is True:
                acc.append(scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind])))
            else:
                acc.append([scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind], knn_dist))])
        return acc
    
    
def knn_cross_val_score_for6(X, y, param, cv, n_folds=3, change='rotate', **kwargs):
    scorer = acc_scorer
    cl = KNNClassifier(**kwargs)
    acc = []
    if change == 'rotate':
        for i in range(len(cv)):
            cl.fit(X[cv[i][0], :], y[(cv[i][0])])
            knn_dist, knn_ind = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
            for p in [param, -param]:
                X_new = np.array([skt.rotate(im.reshape(28, 28), p).ravel() 
                                  for im in X[cv[i][1], :]])
                knn_dist_tmp, knn_ind_tmp = cl.find_kneighbors(X_new, return_distance=True)
                knn_ind[knn_dist > knn_dist_tmp] = knn_ind_tmp[knn_dist > knn_dist_tmp]
                knn_dist[knn_dist > knn_dist_tmp] = knn_dist_tmp[knn_dist > knn_dist_tmp]
                del(X_new)
                del(knn_ind_tmp)
                del(knn_dist_tmp)
            if cl.weights is True:
                acc.append(scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind])))
            else:
                acc.append([scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind], knn_dist))])
        return acc
    if change == 'shift':
        for i in range(len(cv)):
            cl.fit(X[cv[i][0], :], y[(cv[i][0])])
            knn_dist, knn_ind = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
            for p in [param, -param]:
                X_new = np.array([ndimage.shift(im.reshape(28, 28), [p, 0]).ravel()
                                  for im in X[cv[i][1], :]])
                knn_dist_tmp, knn_ind_tmp = cl.find_kneighbors(X_new, return_distance=True)
                knn_ind[knn_dist > knn_dist_tmp] = knn_ind_tmp[knn_dist > knn_dist_tmp]
                knn_dist[knn_dist > knn_dist_tmp] = knn_dist_tmp[knn_dist > knn_dist_tmp]
                del(X_new)
                del(knn_ind_tmp)
                del(knn_dist_tmp)
            for p in [param, -param]:
                X_new = np.array([ndimage.shift(im.reshape(28, 28), [0, p]).ravel()
                                  for im in X[cv[i][1], :]])
                knn_dist_tmp, knn_ind_tmp = cl.find_kneighbors(X_new, return_distance=True)
                knn_ind[knn_dist > knn_dist_tmp] = knn_ind_tmp[knn_dist > knn_dist_tmp]
                knn_dist[knn_dist > knn_dist_tmp] = knn_dist_tmp[knn_dist > knn_dist_tmp]
                del(X_new)
                del(knn_ind_tmp)
                del(knn_dist_tmp)
            if cl.weights is True:
                acc.append(scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind])))
            else:
                acc.append([scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind], knn_dist))])
        return acc
    if change == 'filter':
        for i in range(len(cv)):
            cl.fit(X[cv[i][0], :], y[(cv[i][0])])
            knn_dist, knn_ind = cl.find_kneighbors(X[cv[i][1]], return_distance=True)
            p = param ** (1/2)
            X_new = np.array([filt.gaussian_filter(im.reshape(28, 28), p).ravel()
                              for im in X[cv[i][1], :]])
            knn_dist_tmp, knn_ind_tmp = cl.find_kneighbors(X_new, return_distance=True)
            knn_ind[knn_dist > knn_dist_tmp] = knn_ind_tmp[knn_dist > knn_dist_tmp]
            knn_dist[knn_dist > knn_dist_tmp] = knn_dist_tmp[knn_dist > knn_dist_tmp]
            del(X_new)
            del(knn_ind_tmp)
            del(knn_dist_tmp)
            if cl.weights is True:
                acc.append(scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind])))
            else:
                acc.append([scorer(y[cv[i][1]], cl.predict_k(y[cv[i][0]][knn_ind], knn_dist))])
        return acc
