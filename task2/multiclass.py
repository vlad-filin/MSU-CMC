import optimization as o
import numpy as np


class MulticlassStrategy:   
    def __init__(self, classifier, mode, **kwargs):
        """
        Инициализация мультиклассового классификатора
        
        classifier - базовый бинарный классификатор
        
        mode - способ решения многоклассовой задачи,
        либо 'one_vs_all', либо 'all_vs_all'
        
        **kwargs - параметры классификатор
        """
        self.mode = mode
        self.kwargs = kwargs
        self.classifier = classifier
        
    def fit(self, X, y):
        """
        Обучение классификатора
        
        Y = {0, ..., k}
        """
        self.list_classifier = []
        up = max(y)
        self.up = up
        if self.mode == 'one_vs_all':
            for i in range(0, up + 1):
                self.list_classifier.append(self.classifier(**self.kwargs))
                y_i = np.ones(y.shape)
                y_i[y != i] = -1
                self.list_classifier[i].fit(X, y_i)
        else:
            bottom = min(y)
            self.list_classifier = []
            for j in range(0, up + 1):
                self.list_classifier.append([])
                for s in range(0, j):
                    self.list_classifier[j].append(self.classifier(**self.kwargs))
                    X_sj = X[(y == s) | (y == j)]
                    y_sj = y[(y == s) | (y == j)]
                    y_sj[y_sj == s] = -1
                    y_sj[y_sj == j] = 1
                    (self.list_classifier[j])[s].fit(X_sj, y_sj)
        
    def predict(self, X):
        """
        Выдача предсказаний классификатором
        """
        if self.mode == 'one_vs_all':
            result = []
            for c in self.list_classifier:
                result.append(c.predict_proba(X)[:, 1])
            result = np.array(result)
            return np.argmax(result, axis=0)
        else:
            up = self.up
            result = np.zeros((X.shape[0], up + 1))
            for k in range(0, up + 1):
                cur_sum = np.zeros(X.shape[0])
                for j in range(0, up + 1):
                    for s in range(0, j):
                        res_js = (self.list_classifier[j])[s].predict(X)
                        r = np.zeros(result.shape)
                        tmp = np.zeros(res_js.shape)
                        tmp[res_js == -1] = 1
                        r[:, s] = tmp
                        tmp = np.zeros(res_js.shape)
                        tmp[res_js == 1] = 1
                        r[:, j] = tmp
                        result += r
            return np.argmax(result, axis=1)
