import numpy as np
import scipy
from scipy.special import expit
from scipy.spatial.distance import euclidean
from scipy.special import logsumexp


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов."
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
 
    Оракул должен поддерживать l2 регуляризацию.
    """
 
    def __init__(self, l2_coef=0):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2 = l2_coef

    def func(self, X, y, w0):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        
        return (np.logaddexp(np.zeros(y.shape),
                             -y * (X.dot(w0))).sum() / X.shape[0] +
                self.l2 * 0.5 * (euclidean(w0, np.zeros(w0.shape)) ** 2))
                
    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        if scipy.sparse.issparse(X):
            return np.array(X.T.dot(expit(-y * X.dot(w)) * (-y)) / X.shape[0] + self.l2 * w)
        if (type(y) == type(X)):
            return ((-X * y[:, np.newaxis] *
                     expit(-(y * (X.dot(w))))[:, np.newaxis] /
                     X.shape[0]).sum(axis=0) + self.l2 * w)
        else:
            return ((-X * y *
                     expit(-(y * (X.dot(self.w))))) +
                    self.l2 * w)
    
    
class MulticlassLogistic(BaseSmoothOracle):
    """
    Оракул для задачи многоклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    
    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """
    
    def __init__(self, class_number=0, l2_coef=0):
        """
        Задание параметров оракула.
        
        class_number - количество классов в задаче
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.class_number = class_number
        self.l2 = l2_coef
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - двумерный numpy array
        """
        up = np.amax(y)
        tmp = (y[:, np.newaxis] - np.arange(0, up + 1)[np.newaxis, :]) == 0
        return (((-X.dot(w.transpose()) * tmp).sum(axis=1) +
                 logsumexp(X.dot(w.transpose()), axis=1)).sum()/X.shape[0] +
                self.l2 * (np.diag(w.dot(w.transpose())).sum()) / 2)

    def grad(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - двумерный numpy array
        """
        if self.class_number == 0:
            up = np.amax(y)
        else:
            up = self.class_number - 1
        Xw = X.dot(w.T)
        tmp = (y[:, np.newaxis] - np.arange(0, up + 1)[np.newaxis, :]) == 0
        exp = (np.exp(Xw - np.max(Xw, axis=1)[:, np.newaxis]) /
               np.exp(scipy.special.logsumexp(Xw - 
                                              np.max(Xw, axis=1)[:, np.newaxis], axis=1))[:, np.newaxis])
        return ((1 / X.shape[0]) * (X.T.dot(exp).T - 
                                    X.T.dot(tmp).T) + self.l2 * w)
