import numpy as np
import scipy


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
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

        
class BinaryHinge(BaseSmoothOracle):
    """
    Оракул для задачи двухклассового линейного SVM.
    
    Нулевая координата вектора w соответствует w_0.
    Считается, что в классификатор подаётся X с уже созданным единичным столбцом
    (так делается для того, чтобы не переписывать код из предыдущего задания).
    """
    
    def __init__(self, C=1.0):
        """
        Задание параметров оракула.
        
        C - коэффициент регуляризации в функционале SVM
        """
        self.C = C
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        tmp = 1-(X.dot(w.T))*y
        hinge = np.where(tmp > 0, tmp, np.zeros(tmp.shape))
        return (self.C / X.shape[0]) * np.sum(hinge) + 0.5 * w[1:].dot(w[1:].T)
        
    def grad(self, X, y, w):
        """
        Вычислить субградиент функционала в точке w на выборке X с ответами y.
        Субгрдиент в точке 0 необходимо зафиксировать равным 0.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        ind = np.where(1-X.dot(w.T)*y <= 0)[0]
        tmp = X.copy()
        tmp[ind, :] = 0
        W = w.copy()
        W[0] = 0
        return (self.C) * np.mean( - tmp * y[:,np.newaxis], axis=0) + W