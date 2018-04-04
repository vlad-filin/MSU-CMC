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
        return np.ones((X.shape[0], Y.shape[0])) - X.dot(Y.transpose()) /
    (((X ** 2).sum(axis=1)[:, np.newaxis] ** (1/2)) * 
     ((Y ** 2).sum(axis=1)[np.newaxis, :] ** (1/2)))
        
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